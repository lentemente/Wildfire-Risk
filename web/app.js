/* Wildfire spread risk viewer. Reads data/meta.json, tiles/ and data/*.geojson. */
(async function () {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const num = (v) => Math.round(v).toLocaleString('en-NZ');
  const dur = (min) => (min < 0 ? 'not reached' : min < 60 ? `${Math.round(min)} min` : `${Math.floor(min / 60)} h ${String(Math.round(min % 60)).padStart(2, '0')} min`);

  let meta;
  try {
    meta = await (await fetch('data/meta.json')).json();
  } catch (e) {
    document.body.innerHTML = '<p style="padding:2em">No results found. Run the pipeline first (run.ps1).</p>';
    return;
  }
  const L = meta.legends || {};
  const scen = meta.scenarios || [];
  const hasResults = !!meta.summary;
  const BLUE = ['#b7d3f6', '#6da7ec', '#2a78d6', '#184f95', '#0d366b'];
  const TIME = L.arrival ? L.arrival.colours : [];

  const OVERLAYS = [
    { id: 'burn', label: 'Line fires reaching each spot' },
    { id: 'arrival', label: 'Fire arrival time from the lines' },
    { id: 'ros', label: 'Rate of spread' },
    { id: 'intensity', label: 'Fire intensity' },
    { id: 'fuel', label: 'Fuel type', fixed: true },
    { id: 'canopy', label: 'Vegetation height (LiDAR)', fixed: true },
    { id: 'curing', label: 'Pasture curing (satellite)', fixed: true },
    { id: 'wexp', label: 'Wind exposure', perScenarioOnly: true },
    { id: 'none', label: 'None' },
  ].filter((o) => o.id === 'none' || o.id === 'fuel' ? (o.id === 'none' || meta.fuel_legend) : L[o.id]);

  const state = { scenario: 'worst', overlay: OVERLAYS[0].id, metric: meta.priority ? 'priority' : 'bld', base: 'hillshade' };
  if (!meta.priority) $('line-metric').querySelector('[value=priority]').remove();

  // ---- controls -----------------------------------------------------------
  $('hours').textContent = meta.burn_hours;
  const dirs = { 0: 'N', 45: 'NE', 90: 'E', 135: 'SE', 180: 'S', 225: 'SW', 270: 'W', 315: 'NW' };
  $('scenario').innerHTML = '<option value="worst">Worst of all scenarios</option>' +
    scen.map((s) => `<option value="${s.id}">${esc(s.name)} wind</option>`).join('');
  $('overlays').innerHTML = OVERLAYS.map((o, i) =>
    `<label><input type="radio" name="ov" value="${o.id}" ${i === 0 ? 'checked' : ''}> ${o.label}</label>`).join('');
  if (!window.LINZ_BASEMAPS_KEY) $('aerial-opt').style.display = 'none';

  // ---- map ----------------------------------------------------------------
  const b = meta.bounds;
  const tileBounds = b;
  const style = {
    version: 8,
    sources: {
      hillshade: { type: 'raster', tiles: ['tiles/hillshade/{z}/{x}/{y}.png'], tileSize: 256, minzoom: meta.minzoom, maxzoom: meta.maxzoom, bounds: tileBounds, attribution: 'Elevation: LINZ CC BY 4.0' },
      terrain: { type: 'raster-dem', tiles: ['tiles/terrain/{z}/{x}/{y}.png'], tileSize: 256, encoding: 'terrarium', minzoom: meta.minzoom, maxzoom: meta.maxzoom, bounds: tileBounds },
    },
    layers: [
      { id: 'bg', type: 'background', paint: { 'background-color': '#d8d8d4' } },
      { id: 'hillshade', type: 'raster', source: 'hillshade' },
    ],
    sky: { 'sky-color': '#bcd3e8', 'horizon-color': '#e8eef2', 'fog-color': '#e8eef2', 'sky-horizon-blend': 0.6, 'horizon-fog-blend': 0.6, 'fog-ground-blend': 0.2 },
  };
  if (window.LINZ_BASEMAPS_KEY) {
    style.sources.aerial = { type: 'raster', tiles: [`https://basemaps.linz.govt.nz/v1/tiles/aerial/WebMercatorQuad/{z}/{x}/{y}.webp?api=${window.LINZ_BASEMAPS_KEY}`], tileSize: 256, maxzoom: 18, attribution: 'Imagery: LINZ CC BY 4.0' };
    style.layers.push({ id: 'aerial', type: 'raster', source: 'aerial', layout: { visibility: 'none' } });
  }
  const map = new maplibregl.Map({
    container: 'map', style,
    bounds: [[b[0], b[1]], [b[2], b[3]]],
    fitBoundsOptions: { padding: { left: innerWidth > 640 ? 340 : 10, top: 10, right: 10, bottom: 10 } },
    maxBounds: [[b[0] - 0.6, b[1] - 0.4], [b[2] + 0.6, b[3] + 0.4]],
    minZoom: 6.5, maxZoom: 16, maxPitch: 80, pitch: 50, bearing: -20,
    attributionControl: { compact: true },
  });
  map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), 'top-right');
  map.addControl(new maplibregl.ScaleControl({ unit: 'metric' }), 'bottom-right');
  window.fireMap = map;
  if (location.search.includes('debug')) {
    // ?debug lists map errors on screen (missing tiles are expected and skipped)
    const box = document.body.appendChild(document.createElement('pre'));
    box.style.cssText = 'position:absolute;right:8px;bottom:40px;z-index:9;max-width:50%;max-height:40%;overflow:auto;background:#fff;font-size:11px;padding:6px;white-space:pre-wrap';
    const log = (m) => { box.textContent += m + '\n'; };
    window.addEventListener('error', (e) => log('JS: ' + e.message));
    window.addEventListener('unhandledrejection', (e) => log('Promise: ' + (e.reason && e.reason.message || e.reason)));
    map.on('error', (e) => { const m = String(e.error && e.error.message || e.error); if (!/404|Not Found/.test(m)) log('Map: ' + m); });
    map.on('idle', () => log('idle; layers: ' + map.getStyle().layers.map((l) => l.id).join(',')));
  }
  await new Promise((r) => map.on('load', r));
  map.setTerrain({ source: 'terrain', exaggeration: +$('exag').value });

  // ---- vector layers ------------------------------------------------------
  const FIRST_VECTOR = 'pcl-fill';
  if (hasResults) {
    map.addSource('pcl', { type: 'geojson', data: 'data/pcl.geojson' });
    map.addSource('lines', { type: 'geojson', data: 'data/lines.geojson' });
    map.addSource('bld', { type: 'geojson', data: 'data/buildings.geojson' });
    map.addLayer({ id: 'pcl-fill', type: 'fill', source: 'pcl', paint: { 'fill-color': '#006300', 'fill-opacity': 0.1 } });
    map.addLayer({ id: 'pcl-line', type: 'line', source: 'pcl', paint: { 'line-color': '#006300', 'line-width': 1.2 } });
    map.addLayer({ id: 'lines-case', type: 'line', source: 'lines', layout: { 'line-cap': 'round' }, paint: { 'line-color': '#0b0b0b', 'line-width': 3 } });
    map.addLayer({ id: 'lines', type: 'line', source: 'lines', layout: { 'line-cap': 'round' }, paint: { 'line-color': '#2a78d6', 'line-width': 2 } });
    map.addLayer({ id: 'bld', type: 'circle', source: 'bld', minzoom: 10.5, paint: { 'circle-radius': 3, 'circle-color': '#898781', 'circle-stroke-color': '#ffffff', 'circle-stroke-width': 1 } });
  }
  if (meta.priority) {
    map.addSource('stn', { type: 'geojson', data: 'data/stations.geojson' });
    map.addLayer({ id: 'stn', type: 'circle', source: 'stn', paint: { 'circle-radius': 6, 'circle-color': '#ffffff', 'circle-stroke-color': '#d03b3b', 'circle-stroke-width': 3 } });
    map.on('click', 'stn', (e) => {
      new maplibregl.Popup().setLngLat(e.lngLat).setHTML(`<div class="pop"><h3>${esc(e.features[0].properties.name)}</h3></div>`).addTo(map);
      e.originalEvent.stnHandled = true;
    });
  }
  map.addSource('dist', { type: 'geojson', data: 'data/districts.geojson' });
  map.addLayer({ id: 'dist', type: 'line', source: 'dist', paint: { 'line-color': '#0b0b0b', 'line-width': 1.5, 'line-dasharray': [3, 2] } });
  const firstVector = () => (map.getLayer(FIRST_VECTOR) ? FIRST_VECTOR : 'dist');

  // ---- legends ------------------------------------------------------------
  function swatches(el, title, items, kind) {
    el.innerHTML = (title ? `<div class="t">${esc(title)}</div>` : '') + items.map((it) => {
      const mark = kind === 'line' ? `<span class="ln" style="background:${it.colour};height:${it.w || 4}px"></span>`
        : kind === 'dot' ? `<span class="dot" style="background:${it.colour}"></span>`
          : `<span class="sw" style="background:${it.colour || 'transparent'}"></span>`;
      return `<div class="row">${mark}<span>${esc(it.label)}</span></div>`;
    }).join('');
  }

  function overlayScenario(o) {
    if (o.fixed) return null;
    if (o.perScenarioOnly && state.scenario === 'worst') return scen[0].id;
    return state.scenario;
  }

  function renderOverlay() {
    if (map.getLayer('overlay')) map.removeLayer('overlay');
    if (map.getSource('overlay')) map.removeSource('overlay');
    const o = OVERLAYS.find((x) => x.id === state.overlay);
    const leg = $('legend');
    if (!o || o.id === 'none') { leg.innerHTML = ''; return; }
    const sid = overlayScenario(o);
    const name = sid ? `${o.id}_${sid}` : o.id;
    map.addSource('overlay', { type: 'raster', tiles: [`tiles/${name}/{z}/{x}/{y}.png`], tileSize: 256, minzoom: meta.minzoom, maxzoom: meta.maxzoom, bounds: tileBounds });
    map.addLayer({ id: 'overlay', type: 'raster', source: 'overlay', paint: { 'raster-opacity': $('opacity').value / 100, 'raster-resampling': 'nearest', 'raster-fade-duration': 0 } }, firstVector());
    if (o.id === 'fuel') {
      swatches(leg, 'Fuel type (unshaded = will not burn)', meta.fuel_legend);
    } else {
      const lg = L[o.id];
      let title = `${lg.title} (${lg.unit})`;
      if (o.perScenarioOnly && state.scenario === 'worst') title += `, ${scen[0].name.toLowerCase()} wind`;
      swatches(leg, title, lg.labels.map((label, i) => ({ label, colour: lg.colours[i] })));
      if (lg.note) leg.innerHTML += `<div class="t" style="margin-top:4px">${esc(lg.note)}</div>`;
    }
  }

  const METRIC = {
    bld: { label: 'Buildings reached', unit: '' },
    pcl_ha: { label: 'Conservation land burned', unit: ' ha' },
    ha: { label: 'Area burned', unit: ' ha' },
  };

  function renderVectors() {
    if (!hasResults) return;
    const s = state.scenario;
    if (state.metric === 'priority') {
      // one combined score, banded by share of line length; not scenario-specific
      const P = meta.priority;
      const cols = [BLUE[4], BLUE[3], BLUE[1], BLUE[0]];
      const wds = [5, 3.6, 2.2, 1.4];
      const pick = (arr) => ['match', ['get', 'band_i'], 0, arr[0], 1, arr[1], 2, arr[2], arr[3]];
      map.setPaintProperty('lines', 'line-color', pick(cols));
      map.setPaintProperty('lines', 'line-width', pick(wds));
      map.setPaintProperty('lines-case', 'line-width', ['+', pick(wds), 1.2]);
      map.setLayoutProperty('lines', 'line-sort-key', ['-', 0, ['get', 'band_i']]);
      swatches($('line-legend'), 'Patrol priority (all wind scenarios combined)',
        P.bands.map((b, i) => ({ label: `${b}: ${num(P.km[b])} km, score ${Math.round(P.min_score[b])}+`, colour: cols[i], w: wds[i] + 1 })), 'line');
      renderRest(s);
      return;
    }
    // lines: colour and width step up with the chosen consequence
    const prop = `${state.metric}_${s}`;
    const br = meta.line_breaks[state.metric];
    const colour = ['step', ['get', prop], BLUE[0]];
    const width = ['step', ['get', prop], 1.5];
    br.forEach((v, i) => { colour.push(v, BLUE[i + 1]); width.push(v, 1.5 + (i + 1) * 0.8); });
    map.setPaintProperty('lines', 'line-color', colour);
    map.setPaintProperty('lines', 'line-width', width);
    map.setPaintProperty('lines-case', 'line-width', ['+', width, 1.2]);
    const u = METRIC[state.metric].unit;
    const labels = [`under ${num(br[0])}${u}`];
    br.forEach((v, i) => labels.push(i + 1 < br.length ? `${num(v)} to ${num(br[i + 1])}${u}` : `${num(v)}${u} or more`));
    swatches($('line-legend'), `${METRIC[state.metric].label} by a fire starting on the line`,
      labels.map((label, i) => ({ label, colour: BLUE[i], w: 2 + i * 1.2 })), 'line');
    renderRest(s);
  }

  function renderRest(s) {
    // conservation land: darker where more of it is reached
    map.setPaintProperty('pcl-fill', 'fill-opacity', ['interpolate', ['linear'], ['get', `pct_${s}`], 0, 0.06, 100, 0.55]);

    // buildings: coloured by how soon fire from the network arrives
    const t = ['get', `t_${s}`];
    map.setPaintProperty('bld', 'circle-color', ['case', ['<', t, 0], '#c3c2b7',
      ['step', t, TIME[0], 60, TIME[1], 120, TIME[2], 240, TIME[3]]]);
    map.setPaintProperty('bld', 'circle-radius', ['case', ['<', t, 0], 2.2, ['==', ['get', 'c'], 1], 4.5, 3.2]);
    swatches($('bld-legend'), 'Buildings: time until fire from a line arrives (larger dot = commercial or industrial area)', [
      { label: 'under 1 hour', colour: TIME[0] }, { label: '1 to 2 hours', colour: TIME[1] },
      { label: '2 to 4 hours', colour: TIME[2] }, { label: `4 to ${meta.burn_hours} hours`, colour: TIME[3] },
      { label: 'not reached', colour: '#c3c2b7' }], 'dot');
  }

  function renderNote() {
    const s = scen.find((x) => x.id === state.scenario);
    $('scenario-note').textContent = s
      ? `${s.speed_kmh} km/h from the ${dirs[s.from_deg] || s.from_deg + '°'}, ${s.temp_c} °C, ${s.rh_pct}% humidity. ${Math.round(s.share * 100)}% of strong-wind hours on hot, dry afternoons.`
      : 'Highest value across the scenarios below, cell by cell.';
  }

  function renderSummary() {
    if (!hasResults) { $('summary').innerHTML = '<p class="note">Simulation results not built yet.</p>'; return; }
    const m = meta.summary;
    const rows = [
      ['Power lines assessed', `${num(m.line_km)} km`],
      ['Simulated ignitions', num(m.ignitions)],
    ];
    const bs = m.by_scenario[state.scenario];
    if (bs) {
      rows.push(['Typical fire size (median)', `${num(bs.median_ha)} ha`], ['Large fire (1 in 10)', `${num(bs.p90_ha)} ha`],
        ['Fires reaching a building', `${Math.round(bs.share_reaching_building * 100)}%`],
        ['Fires reaching conservation land', `${Math.round(bs.share_reaching_pcl * 100)}%`]);
    } else {
      rows.push(['Buildings fire could reach', `${num(m.buildings_reached_worst)} of ${num(m.buildings)}`],
        ['of which in business areas', num(m.business_buildings_reached_worst)],
        ['Conservation land reachable', `${num(m.pcl_ha_reached_worst)} of ${num(m.pcl_ha)} ha`]);
    }
    $('summary').innerHTML = '<span class="h">Summary</span><table>' +
      rows.map((r) => `<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`).join('') + '</table>';
  }

  function renderAll() { renderNote(); renderOverlay(); renderVectors(); renderSummary(); }
  renderAll();

  // ---- events -------------------------------------------------------------
  $('scenario').onchange = (e) => { state.scenario = e.target.value; renderAll(); };
  $('overlays').onchange = (e) => { state.overlay = e.target.value; renderOverlay(); };
  $('line-metric').onchange = (e) => { state.metric = e.target.value; renderVectors(); };
  $('opacity').oninput = (e) => { if (map.getLayer('overlay')) map.setPaintProperty('overlay', 'raster-opacity', e.target.value / 100); };
  $('exag').oninput = (e) => {
    const v = +e.target.value;
    map.setTerrain(v > 0 ? { source: 'terrain', exaggeration: v } : null);
  };
  $('basemaps').onchange = (e) => {
    const aerial = e.target.value === 'aerial';
    if (map.getLayer('aerial')) map.setLayoutProperty('aerial', 'visibility', aerial ? 'visible' : 'none');
    map.setLayoutProperty('hillshade', 'visibility', aerial ? 'none' : 'visible');
  };
  const vis = (ids, on) => ids.forEach((id) => map.getLayer(id) && map.setLayoutProperty(id, 'visibility', on ? 'visible' : 'none'));
  $('t-lines').onchange = (e) => vis(['lines', 'lines-case'], e.target.checked);
  $('t-pcl').onchange = (e) => vis(['pcl-fill', 'pcl-line'], e.target.checked);
  $('t-bld').onchange = (e) => vis(['bld'], e.target.checked);
  $('t-stn').onchange = (e) => vis(['stn'], e.target.checked);
  $('t-dist').onchange = (e) => vis(['dist'], e.target.checked);
  $('toggle-panel').onclick = () => $('panel').classList.toggle('hidden');

  // ---- popups -------------------------------------------------------------
  const table = (rows) => '<table>' + rows.map((r) => `<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`).join('') + '</table>';
  const scenLabel = () => (state.scenario === 'worst' ? 'worst of all scenarios' : `${scen.find((x) => x.id === state.scenario).name.toLowerCase()} wind`);
  function popup(lngLat, html) {
    new maplibregl.Popup({ maxWidth: '300px' }).setLngLat(lngLat).setHTML(`<div class="pop">${html}<div class="k">${esc(scenLabel())}</div></div>`).addTo(map);
  }
  if (hasResults) {
    map.on('click', (e) => {
      if (e.originalEvent && e.originalEvent.stnHandled) return;
      const s = state.scenario;
      const pad = 5;
      const box = [[e.point.x - pad, e.point.y - pad], [e.point.x + pad, e.point.y + pad]];
      const layers = ['bld', 'lines', 'pcl-fill'].filter((id) => map.getLayer(id) && map.getLayoutProperty(id, 'visibility') !== 'none');
      const f = map.queryRenderedFeatures(box, { layers })[0];
      if (!f) return;
      const p = f.properties;
      if (f.layer.id === 'lines') {
        const pri = meta.priority && p.score !== undefined ? `<div class="k">Patrol priority</div>` + table([
          ['<b>Score</b>', `<b>${(+p.score).toFixed(1)} of 100</b> (${esc(p.band)}, rank ${num(p.rank)})`],
          ['Property', (+p.s_property).toFixed(0)], ['Conservation', (+p.s_conservation).toFixed(0)],
          ['Response', (+p.s_response).toFixed(0)], ['Spread', (+p.s_spread).toFixed(0)],
          ['First fire crew arrives after', dur(p.resp_min)],
          ['Fire size by then', `${num(p.ha_arr)} ha`],
          ['Buildings reached by then', num(p.bld_arr)]]) : '';
        popup(e.lngLat, `<h3>${p.voltage ? p.voltage / 1000 + ' kV' : ''} power line</h3>` + pri +
          `<div class="k">If a fire started on this section and burned for ${meta.burn_hours} hours:</div>` + table([
          ['Area burned', `${num(p[`ha_${s}`])} ha`],
          ['Conservation land burned', `${num(p[`pcl_ha_${s}`])} ha`],
          ['Buildings reached', num(p[`bld_${s}`])],
          ['of which in business areas', num(p[`bldc_${s}`])],
          ['Time to first building', dur(p[`tfirst_${s}`])]]));
      } else if (f.layer.id === 'bld') {
        popup(e.lngLat, `<h3>Building${p.c ? ' (commercial or industrial area)' : ''}</h3>` + table([
          ['Footprint', `${num(p.m2)} m²`],
          ['Simulated line fires that reach it', num(p[`n_${s}`])],
          ['Soonest arrival from a line', dur(p[`t_${s}`])]]));
      } else {
        popup(e.lngLat, `<h3>${esc(p.name || 'Conservation land')}</h3><div class="k">${esc(p.type || '')}</div>` + table([
          ['Area in study', `${num(p.ha)} ha`],
          ['Reachable by a line fire', `${num(p[`pct_${s}`])}%`],
          ['Soonest arrival from a line', dur(p[`t_${s}`])]]));
      }
    });
    ['bld', 'lines', 'pcl-fill'].forEach((id) => {
      map.on('mouseenter', id, () => { map.getCanvas().style.cursor = 'pointer'; });
      map.on('mouseleave', id, () => { map.getCanvas().style.cursor = ''; });
    });
  }
})();
