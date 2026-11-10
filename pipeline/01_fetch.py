"""Download and cache every input, clipped to the analysis rectangle.

Usage:  python pipeline/01_fetch.py [step ...]
Steps:  extent dem pcl osm stations weather lines wfs     (default: all)
Each step is skipped if its output already exists; delete the file to refetch.
"""
from __future__ import annotations

import io
import json
import re
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import LineString, MultiLineString, Polygon, box, shape
from shapely.ops import polygonize, unary_union

import firelib as fl

UA = {"User-Agent": "wildfire-risk-pipeline/0.1"}
CFG = fl.load_config()
RAW = fl.RAW
RAW.mkdir(parents=True, exist_ok=True)


def get(url, tries=5, **kw):
    kw.setdefault("timeout", 180)
    kw.setdefault("headers", UA)
    for i in range(tries):
        try:
            r = requests.get(url, **kw)
            if r.status_code in (429, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}")
            return r
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            if i == tries - 1:
                raise
            wait = 5 * 2 ** i
            print(f"    retry in {wait}s ({type(e).__name__})")
            time.sleep(wait)


def extent() -> dict:
    return json.loads((RAW / "extent.json").read_text())


# ---------------------------------------------------------------------------
def step_extent():
    out = RAW / "districts.gpkg"
    if out.exists() and (RAW / "extent.json").exists():
        return
    feats = []
    for name in CFG["districts"]:
        r = get("https://nominatim.openstreetmap.org/search",
                params={"q": name + ", New Zealand", "format": "geojson",
                        "polygon_geojson": 1, "limit": 5})
        cands = [f for f in r.json()["features"]
                 if f["properties"].get("type") == "administrative"
                 and f["geometry"]["type"] in ("Polygon", "MultiPolygon")]
        if not cands:
            raise SystemExit(f"District boundary not found for {name!r}")
        feats.append({"name": name, "geometry": shape(cands[0]["geometry"])})
        time.sleep(1.1)
    gdf = gpd.GeoDataFrame(feats, crs="EPSG:4326").to_crs(2193)
    gdf.to_file(out, driver="GPKG")
    b = gdf.total_bounds
    buf, snap = CFG["buffer_m"], 1000.0
    ext = {
        "xmin": float(np.floor((b[0] - buf) / snap) * snap),
        "ymin": float(np.floor((b[1] - buf) / snap) * snap),
        "xmax": float(np.ceil((b[2] + buf) / snap) * snap),
        "ymax": float(np.ceil((b[3] + buf) / snap) * snap),
    }
    (RAW / "extent.json").write_text(json.dumps(ext))
    print("  extent", ext, "area km2:", [round(a / 1e6) for a in gdf.area])


# ---------------------------------------------------------------------------
_ROWS = [a + b for a in "ABC" for b in "ABCDEFGHJKLMNPQRSTUVWXYZ"]
_ROWS = _ROWS[_ROWS.index("AS"):]          # Topo50 rows run AS, AT, ... (no I or O)


def step_dem():
    """LINZ 8 m DEM (contour-derived, complete coverage)."""
    mosaic_sheets(CFG["dem_url"], RAW / "dem.tif")


def step_lidar():
    """LINZ 1 m LiDAR ground (DEM) and surface (DSM) models, where flown."""
    mosaic_sheets(CFG["lidar_dem_url"], RAW / "lidar_dem.tif")
    mosaic_sheets(CFG["lidar_dsm_url"], RAW / "lidar_dsm.tif")


def mosaic_sheets(base_url, out):
    """Mosaic Topo50-sheet COGs to the analysis cell size, straight from S3."""
    import rasterio
    from rasterio.enums import Resampling
    if out.exists():
        return
    e, cell = extent(), CFG["cell_m"]
    ncol = int((e["xmax"] - e["xmin"]) / cell)
    nrow = int((e["ymax"] - e["ymin"]) / cell)
    dem = np.full((nrow, ncol), np.nan, dtype=np.float32)
    sw, sh = 24000.0, 36000.0
    c0, c1 = int((e["xmin"] - 988000) // sw), int((e["xmax"] - 1 - 988000) // sw)
    r0, r1 = int((6234000 - e["ymax"]) // sh), int((6234000 - e["ymin"] - 1) // sh)
    sheets = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]

    def load(rc):
        r, c = rc
        name = f"{_ROWS[r]}{c:02d}"
        url = f"{base_url}/{name}.tiff"
        for attempt in range(4):
            try:
                with rasterio.open(url) as src:
                    a = src.read(1, out_shape=(int(sh / cell), int(sw / cell)),
                                 resampling=Resampling.average, masked=True)
                    return rc, name, a.filled(np.nan).astype(np.float32)
            except rasterio.errors.RasterioIOError as ex:
                if "404" in str(ex) or "does not exist" in str(ex):
                    return rc, name, None       # sea / no sheet
                time.sleep(3 * (attempt + 1))
        raise SystemExit(f"Could not read DEM sheet {name}")

    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="3"):
        with ThreadPoolExecutor(8) as pool:
            for (r, c), name, a in pool.map(load, sheets):
                if a is None:
                    continue
                x0, y1 = 988000 + c * sw, 6234000 - r * sh
                col = int(round((x0 - e["xmin"]) / cell))
                row = int(round((e["ymax"] - y1) / cell))
                rs, cs = max(row, 0), max(col, 0)
                re_, ce = min(row + a.shape[0], nrow), min(col + a.shape[1], ncol)
                dem[rs:re_, cs:ce] = a[rs - row:re_ - row, cs - col:ce - col]
    from affine import Affine
    with rasterio.open(out, "w", driver="GTiff", height=nrow, width=ncol, count=1,
                       dtype="float32", crs="EPSG:2193", nodata=np.nan,
                       transform=Affine(cell, 0, e["xmin"], 0, -cell, e["ymax"]),
                       compress="deflate", predictor=3, tiled=True,
                       BIGTIFF="IF_SAFER") as dst:
        dst.write(dem, 1)
    print(f"  {out.name}: {ncol} x {nrow}, covers {1 - np.isnan(dem).mean():.1%}, "
          f"range {np.nanmin(dem):.0f}..{np.nanmax(dem):.0f} m")


# ---------------------------------------------------------------------------
def step_ndvi():
    """Summer greenness (NDVI) from Sentinel-2: median of the clearest scenes."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
    from affine import Affine
    out = RAW / "ndvi.tif"
    if out.exists():
        return
    e, cell, sat = extent(), CFG["cell_m"], CFG["satellite"]
    ll = gpd.GeoSeries([box(e["xmin"], e["ymin"], e["xmax"], e["ymax"])], crs=2193).to_crs(4326).total_bounds
    items = []
    for yr in sat["years"]:
        body = {"collections": ["sentinel-2-l2a"], "bbox": [float(v) for v in ll],
                "datetime": f"{yr}-{sat['season'][0]}T00:00:00Z/{yr}-{sat['season'][1]}T23:59:59Z",
                "limit": 100, "query": {"eo:cloud_cover": {"lt": sat["max_cloud_pct"]}}}
        url = "https://earth-search.aws.element84.com/v1/search"
        while url:
            r = requests.post(url, json=body, headers=UA, timeout=120)
            if not r.ok:
                raise SystemExit(f"Sentinel-2 search failed: HTTP {r.status_code} {r.text[:200]}")
            js = r.json()
            items += js["features"]
            nxt = [l for l in js.get("links", []) if l["rel"] == "next"]
            url, body = (nxt[0]["href"], nxt[0].get("body", body)) if nxt else (None, None)
    by_tile = {}
    for it in items:
        by_tile.setdefault(it["properties"]["grid:code"], []).append(it)
    print(f"  {len(items)} scenes over {len(by_tile)} tiles")

    ncol = int((e["xmax"] - e["xmin"]) / cell)
    nrow = int((e["ymax"] - e["ymin"]) / cell)
    tr = Affine(cell, 0, e["xmin"], 0, -cell, e["ymax"])
    total = np.zeros((nrow, ncol), dtype=np.float32)
    count = np.zeros((nrow, ncol), dtype=np.uint8)
    n = 2745                                      # 40 m, read from overviews

    def scene(it):
        p, a = it["properties"], it["assets"]
        for attempt in range(4):
            try:
                with rasterio.open(a["red"]["href"]) as s:
                    red = s.read(1, out_shape=(n, n), resampling=Resampling.average).astype(np.float32)
                    crs, bounds = s.crs, s.bounds
                with rasterio.open(a["nir"]["href"]) as s:
                    nir = s.read(1, out_shape=(n, n), resampling=Resampling.average).astype(np.float32)
                with rasterio.open(a["scl"]["href"]) as s:
                    scl = s.read(1, out_shape=(n, n), resampling=Resampling.nearest)
                break
            except rasterio.errors.RasterioIOError:
                if attempt == 3:
                    return None
                time.sleep(5 * (attempt + 1))
        ok = (red > 0) & (nir > 0) & np.isin(scl, (4, 5, 6, 7))     # vegetation, bare, water, unclassified
        # Newer processing adds 1000 to every value. The catalogue's flag for
        # whether that has been removed is unreliable, so test the data: with
        # the offset still present, even the darkest pixels sit near 1000.
        if ok.any() and np.percentile(red[ok][::50], 1) > 900:
            red, nir = red - 1000, nir - 1000
            ok &= (red > 0) & (nir > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            v = (nir - red) / (nir + red)
        v[~ok | ~np.isfinite(v)] = np.nan
        return v, crs, bounds

    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="3", AWS_NO_SIGN_REQUEST="YES"):
        for code, its in sorted(by_tile.items()):
            its = sorted(its, key=lambda i: i["properties"]["eo:cloud_cover"])[: sat["scenes_per_tile"]]
            with ThreadPoolExecutor(4) as pool:
                res = [x for x in pool.map(scene, its) if x is not None]
            if not res:
                continue
            with np.errstate(all="ignore"):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # percentile across scenes, ignoring gaps: sort (NaN last),
                    # then pick the rank scaled to each pixel's valid count
                    st = np.sort(np.stack([x[0] for x in res]), axis=0)
                    nvalid = np.isfinite(st).sum(axis=0)
                    k = np.clip(np.round((nvalid - 1) * sat["ndvi_percentile"] / 100.0), 0, None).astype(np.int64)
                    med = np.take_along_axis(st, k[None], axis=0)[0]
                    med[nvalid == 0] = np.nan
                    del st
            crs, b = res[0][1], res[0][2]
            dst = np.full((nrow, ncol), np.nan, dtype=np.float32)
            reproject(med, dst, src_transform=rasterio.transform.from_bounds(*b, n, n), src_crs=crs,
                      src_nodata=np.nan, dst_transform=tr, dst_crs="EPSG:2193", dst_nodata=np.nan,
                      resampling=Resampling.bilinear, num_threads=8)
            ok = np.isfinite(dst)
            total[ok] += dst[ok]
            count[ok] += 1
            print(f"  {code}: {len(res)} scenes")
    ndvi = np.where(count > 0, total / np.maximum(count, 1), np.nan).astype(np.float32)
    with rasterio.open(out, "w", driver="GTiff", height=nrow, width=ncol, count=1,
                       dtype="float32", crs="EPSG:2193", nodata=np.nan, transform=tr,
                       compress="deflate", predictor=3, tiled=True) as d:
        d.write(ndvi, 1)
    print(f"  ndvi covers {np.isfinite(ndvi).mean():.1%}, median {np.nanmedian(ndvi[::9, ::9]):.2f}")


# ---------------------------------------------------------------------------
def step_pcl():
    out = RAW / "pcl.gpkg"
    if out.exists():
        return
    e = extent()
    parts, offset = [], 0
    while True:
        r = get(CFG["pcl_url"] + "/query", params={
            "where": "1=1", "geometry": f"{e['xmin']},{e['ymin']},{e['xmax']},{e['ymax']}",
            "geometryType": "esriGeometryEnvelope", "inSR": 2193,
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "OBJECTID,Type,Name,Legislation", "outSR": 4326,
            "orderByFields": "OBJECTID", "f": "geojson",
            "resultOffset": offset, "resultRecordCount": 1000})
        js = r.json()
        feats = js.get("features", [])
        if not feats:
            if "error" in js:
                raise SystemExit(f"DOC service error: {js['error']}")
            break
        parts.append(gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326"))
        offset += len(feats)
        print("  pcl", offset)
        if len(feats) < 1000:
            break
    gdf = pd.concat(parts, ignore_index=True).to_crs(2193)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]
    gdf.to_file(out, driver="GPKG")


# ---------------------------------------------------------------------------
def step_osm():
    """Commercial / industrial / retail land-use polygons from OpenStreetMap."""
    out = RAW / "osm_business.gpkg"
    if out.exists():
        return
    e = extent()
    ll = gpd.GeoSeries([box(e["xmin"], e["ymin"], e["xmax"], e["ymax"])], crs=2193).to_crs(4326).total_bounds
    bbox = f"{ll[1]},{ll[0]},{ll[3]},{ll[2]}"
    sel = '["landuse"~"^(commercial|industrial|retail)$"]'
    q = f"[out:json][timeout:180];(way{sel}({bbox});relation{sel}({bbox}););out geom;"
    js = None
    for attempt in range(5):
        try:
            r = requests.post("https://overpass-api.de/api/interpreter",
                              data={"data": q}, headers=UA, timeout=300)
            if r.ok:
                js = r.json()
                break
            print(f"    Overpass HTTP {r.status_code}, retrying")
        except requests.RequestException as ex:
            print(f"    Overpass {type(ex).__name__}, retrying")
        time.sleep(20 * (attempt + 1))
    if js is None:
        raise SystemExit("Overpass request failed")
    rows = []
    for el in js["elements"]:
        lu = el.get("tags", {}).get("landuse")
        if el["type"] == "way":
            pts = [(p["lon"], p["lat"]) for p in el.get("geometry", [])]
            if len(pts) >= 4 and pts[0] == pts[-1]:
                rows.append({"landuse": lu, "geometry": Polygon(pts)})
        else:
            lines = [LineString([(p["lon"], p["lat"]) for p in m["geometry"]])
                     for m in el.get("members", [])
                     if m["type"] == "way" and m.get("role") != "inner" and len(m.get("geometry", [])) >= 2]
            for poly in polygonize(unary_union(lines)) if lines else []:
                rows.append({"landuse": lu, "geometry": poly})
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326").to_crs(2193)
    gdf = gdf[gdf.geometry.is_valid]
    gdf.to_file(out, driver="GPKG")
    print("  osm business polygons:", len(gdf))


# ---------------------------------------------------------------------------
def step_stations():
    """Fire stations from OpenStreetMap (airport rescue stations excluded)."""
    out = RAW / "stations.gpkg"
    if out.exists():
        return
    e = extent()
    ll = gpd.GeoSeries([box(e["xmin"], e["ymin"], e["xmax"], e["ymax"])], crs=2193).to_crs(4326).total_bounds
    q = (f'[out:json][timeout:120];nwr["amenity"="fire_station"]'
         f'({ll[1]},{ll[0]},{ll[3]},{ll[2]});out center tags;')
    r = None
    for attempt in range(5):
        try:
            r = requests.post("https://overpass-api.de/api/interpreter",
                              data={"data": q}, headers=UA, timeout=200)
            if r.ok:
                break
        except requests.RequestException:
            pass
        time.sleep(20 * (attempt + 1))
    if r is None or not r.ok:
        raise SystemExit("Overpass request for fire stations failed")
    rows = []
    for el in r.json()["elements"]:
        name = el.get("tags", {}).get("name", "Fire station")
        if "airport" in name.lower():
            continue
        c = el.get("center", el)
        rows.append({"name": name, "lon": c["lon"], "lat": c["lat"]})
    df = pd.DataFrame(rows)
    gdf = gpd.GeoDataFrame(df[["name"]], geometry=gpd.points_from_xy(df.lon, df.lat),
                           crs=4326).to_crs(2193)
    gdf = gdf[gdf.within(box(e["xmin"], e["ymin"], e["xmax"], e["ymax"]))]
    gdf.to_file(out, driver="GPKG")
    print("  fire stations in the area:", len(gdf))


# ---------------------------------------------------------------------------
def step_weather():
    out = RAW / "weather.parquet"
    out = out.with_suffix(".csv.gz")
    if out.exists():
        return
    e, w = extent(), CFG["weather"]
    nx, ny = w["grid_points"]
    xs = e["xmin"] + (np.arange(nx) + 0.5) * (e["xmax"] - e["xmin"]) / nx
    ys = e["ymin"] + (np.arange(ny) + 0.5) * (e["ymax"] - e["ymin"]) / ny
    pts = gpd.GeoSeries(gpd.points_from_xy(*[a.ravel() for a in np.meshgrid(xs, ys)]), crs=2193).to_crs(4326)
    cache = RAW / "weather"
    cache.mkdir(exist_ok=True)
    frames = []
    for i, p in enumerate(pts):
        for yr in range(w["start_year"], w["end_year"] + 1):
            f = cache / f"p{i:02d}_{yr}.json"
            if not f.exists():
                r = get("https://archive-api.open-meteo.com/v1/archive", params={
                    "latitude": round(p.y, 3), "longitude": round(p.x, 3),
                    "start_date": f"{yr}-11-01", "end_date": f"{yr + 1}-03-31",
                    "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m",
                    "models": "era5", "timezone": "Pacific/Auckland"})
                if not r.ok:
                    raise SystemExit(f"Open-Meteo error {r.status_code}: {r.text[:200]}")
                f.write_text(r.text)
                time.sleep(0.4)
            h = json.loads(f.read_text())["hourly"]
            df = pd.DataFrame(h)
            df["point"] = i
            frames.append(df)
        print(f"  weather point {i + 1}/{len(pts)}")
    pd.concat(frames, ignore_index=True).to_csv(out, index=False)


# ---------------------------------------------------------------------------
def step_lines():
    """Power lines from the KMZ that touch the study districts."""
    out = RAW / "lines.gpkg"
    if out.exists():
        return
    kml = None
    with zipfile.ZipFile(fl.ROOT / CFG["kmz"]) as z:
        for n in z.namelist():
            if n.lower().endswith(".kml"):
                kml = z.read(n).decode("utf-8", "replace")
    rows = []
    for pm in re.finditer(r"<Placemark\b.*?</Placemark>", kml, re.S):
        s = pm.group(0)
        v = re.search(r"<td>NOMINALVOLTAGE</td>\s*<td>(.*?)</td>", s)
        volt = int(v.group(1)) if v and v.group(1).isdigit() else 0
        parts = []
        for cm in re.finditer(r"<coordinates>(.*?)</coordinates>", s, re.S):
            pts = [tuple(map(float, t.split(",")[:2])) for t in cm.group(1).split()]
            if len(pts) >= 2:
                parts.append(LineString(pts))
        if parts:
            rows.append({"voltage": volt,
                         "geometry": parts[0] if len(parts) == 1 else MultiLineString(parts)})
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326").to_crs(2193)
    total = len(gdf)
    dist = gpd.read_file(RAW / "districts.gpkg").union_all()
    gdf = gdf[gdf.intersects(dist)].explode(index_parts=False).reset_index(drop=True)
    # cut long lines into patrol-sized sections so each gets its own result
    from shapely.ops import substring
    max_len = float(CFG["simulation"].get("section_max_m", 1000))
    rows = []
    for volt, geom in zip(gdf.voltage, gdf.geometry):
        n = max(1, int(np.ceil(geom.length / max_len)))
        for i in range(n):
            rows.append({"voltage": volt, "geometry": substring(
                geom, i / n, (i + 1) / n, normalized=True) if n > 1 else geom})
    gdf = gpd.GeoDataFrame(rows, crs=2193)
    gdf = gdf[gdf.length > 0].reset_index(drop=True)
    gdf["line_id"] = np.arange(len(gdf))
    gdf["length_m"] = gdf.length.round(1)
    gdf.to_file(out, driver="GPKG")
    print(f"  lines: {len(gdf)} of {total} in the districts, {gdf.length.sum() / 1000:.0f} km")


# ---------------------------------------------------------------------------
def wfs(host, key, layer_id, e, page=2000):
    """Fetch one Koordinates layer inside the extent, as EPSG:2193."""
    base = f"https://{host}/services;key={key}/wfs"
    bbox = f"{e['xmin']},{e['ymin']},{e['xmax']},{e['ymax']},EPSG:2193"
    parts, start = [], 0
    while True:
        r = get(base, params={
            "service": "WFS", "version": "2.0.0", "request": "GetFeature",
            "typeNames": f"layer-{layer_id}", "srsName": "EPSG:2193",
            "outputFormat": "json", "bbox": bbox,
            "count": page, "startIndex": start})
        if r.status_code in (401, 403):
            raise SystemExit(f"{host} rejected the API key (HTTP {r.status_code}). "
                             "Check the key in .env and that it has WFS access.")
        if not r.ok or not r.text.lstrip().startswith("{"):
            raise SystemExit(f"{host} layer {layer_id}: HTTP {r.status_code} {r.text[:300]}")
        feats = r.json().get("features", [])
        if not feats:
            break
        parts.append(gpd.read_file(io.BytesIO(r.content)))
        start += len(feats)
        print(f"    {start} features")
        if len(feats) < page:
            break
    if not parts:
        return gpd.GeoDataFrame(geometry=[], crs=2193)
    gdf = pd.concat(parts, ignore_index=True)
    gdf = gdf.set_crs(2193, allow_override=True)
    # guard against an axis-order mix-up or an ignored bbox
    inside = gdf.intersects(box(e["xmin"], e["ymin"], e["xmax"], e["ymax"]))
    if inside.mean() < 0.95:
        raise SystemExit(f"{host} layer {layer_id}: only {inside.mean():.0%} of the returned "
                         f"features touch the requested area (bounds {gdf.total_bounds.round()}).")
    return gdf[inside].reset_index(drop=True)


def step_wfs():
    env, e = fl.load_env(), extent()
    missing = sorted({v["key"] for v in CFG["layers"].values()
                      if not v.get("optional") and v["key"] not in env})
    if missing:
        raise SystemExit("Missing API key(s) in .env: " + ", ".join(missing))
    for name, v in CFG["layers"].items():
        out = RAW / f"{name}.gpkg"
        if out.exists():
            continue
        if v["key"] not in env:
            print(f"  skipping optional layer {name} (no {v['key']})")
            continue
        print(f"  {name} (layer {v['id']})")
        gdf = wfs(v["host"], env[v["key"]], v["id"], e)
        gdf.to_file(out, driver="GPKG")
        print(f"  {name}: {len(gdf)} features")


STEPS = {"extent": step_extent, "dem": step_dem, "lidar": step_lidar, "ndvi": step_ndvi, "pcl": step_pcl, "osm": step_osm,
         "stations": step_stations, "weather": step_weather, "lines": step_lines, "wfs": step_wfs}

if __name__ == "__main__":
    for s in (sys.argv[1:] or list(STEPS)):
        print(f"[{s}]")
        STEPS[s]()
    print("done")
