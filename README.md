# Wildfire spread risk map: Central Otago and Queenstown Lakes

An interactive 3D map of where a fire starting on a power line would spread, and what it would reach: public conservation land, homes and businesses.

This is a scenario-based screening model. It is not a fire forecast and does not model firefighting.

## Running it

1. Put your API keys in `.env` (see below).
2. In PowerShell, from this folder: `.\run.ps1`
   It downloads the data, runs the model and opens the map at http://localhost:8765.
3. To reopen the map later without rebuilding: `.\run.ps1 -Serve`

Downloads are cached in `data/raw`. Delete a file there to fetch it again.

### API keys (`.env`)

| Line in `.env` | Where to get it | Needed for |
|---|---|---|
| `LINZ_API_KEY` | data.linz.govt.nz → profile → API keys (WFS access) | Buildings, rivers, lakes, roads |
| `LRIS_API_KEY` | lris.scinfo.org.nz → profile → API keys (WFS access) | Land cover |
| `LINZ_BASEMAPS_KEY` (optional) | basemaps.linz.govt.nz | Aerial imagery in the viewer |
| `MFE_API_KEY` (optional) | data.mfe.govt.nz | Irrigated land |

## How the model works

| Step | Script | What it does |
|---|---|---|
| 1 | `pipeline/01_fetch.py` | Downloads inputs for the two districts plus a 5 km buffer |
| 2 | `pipeline/02_grids.py` | Builds 25 m grids of terrain, fuel, barriers, conservation land and buildings |
| 3 | `pipeline/03_wind.py` | Picks three wind scenarios from 20 summers of weather records and works out how terrain shelters or exposes each cell |
| 4 | `pipeline/04_spread.py` | Calculates how fast and in which direction fire would spread from each cell |
| 5 | `pipeline/05_simulate.py` | Starts a fire about every 500 m along the power lines, grows each for 6 hours, and tallies what it burns |
| 6 | `pipeline/06_export.py` | Writes map tiles and layers for the viewer |

**Fuel.** Each land cover class is assigned a fuel type and spread model in `fuels.csv`. Edit that file to change assumptions such as how cured the grass is.

**LiDAR.** Where LINZ has flown LiDAR (about half the study rectangle), its 1 m ground model replaces the contour-derived terrain. Vegetation height is the LiDAR surface model minus the ground model, averaged over each 25 m cell. Pasture or tussock cells averaging 1 m or taller are treated as scrub, and 5 m or taller as conifer trees. This picks up shelterbelts, wilding conifers and scrub invasion that the land cover map misses.

**Satellite.** Sentinel-2 summer greenness (NDVI, January to mid-March over the last two summers, drier end of the clear scenes) sets how cured pasture and crops are in each cell: NDVI 0.25 or less is fully cured, 0.75 or more is too green to burn. This replaces a fixed curing figure per land cover class and identifies irrigated paddocks directly. Tussock is left at fully cured, as the manual assumes.

**Wind.** Scenarios are the three most common strong-wind directions on hot, dry summer afternoons, each with its 95th-percentile wind speed. Wind is increased on exposed ridges and reduced in the lee of terrain.

**Rate of spread.** Uses the models named in Scion's *A Manual for Predicting Fire Behaviour in New Zealand Fuels* (Pearce & Anderson 2008): Canadian grass models for pasture and tussock, the New Zealand scrubland model, and Canadian forest models for conifer and indigenous forest. Slope is converted to an equivalent upslope wind and added to the real wind, so fire runs fastest where wind and slope line up.

**Barriers.** Lakes, wide rivers, rock, gravel, snow and built-up areas stop fire. Streams and sealed roads slow it.

**Fire growth.** Each fire spreads as an ellipse stretched downwind, following the fastest route across the grid.

**Buildings.** A building counts as reached when fire burns ground within 50 m of it.

Settings are in `config.yaml`. Tests for the fire equations and the growth model: `.venv\Scripts\python -m pytest tests`.

## Patrol priority score

Each line section (at most 1 km long) gets one score from 0 to 100, used to decide where to patrol for defects before the fire season. It measures consequence if a fire starts there. It does not use fault history or asset condition.

| Part | Weight | What it measures |
|---|---|---|
| Property | 40% | Buildings the fire reaches. A building in a business area counts three times |
| Conservation | 25% | Hectares of public conservation land burned |
| Response | 25% | Hectares already burning when the first fire crew arrives |
| Spread | 10% | Total hectares burned in 6 hours |

- Results are averaged across the wind scenarios, weighted by how often each occurs.
- Each part is scaled 0 to 100 on a log scale against the 99th-percentile ignition point, so a section must be roughly ten times worse to score twice as high.
- A section takes the score of its worst ignition point.
- Sections are ranked and banded by share of line length: Very high (top 10%), High (next 20%), Medium (next 30%), Low.
- Crew arrival time is an 8-minute turnout plus travel from the nearest fire station (OpenStreetMap) along roads at 70, 40 or 25 km/h by surface, then 4 km/h across open ground. Stations outside the study rectangle are ignored, so arrival times near its edge may be overstated.

Weights, the business multiplier, band sizes and response speeds are in `config.yaml` under `priority` and `response`. `pipeline/02b_response.py` builds the arrival-time grid and `pipeline/05b_priority.py` the scores. The ranked list is written to `outputs/patrol_priority.csv`.

## Map layers

- **Line fires reaching each spot**: how many of the simulated fires burn over each cell.
- **Fire arrival time**: how soon fire from the nearest line could arrive.
- **Rate of spread** and **fire intensity**: how a fire would behave at each cell, wherever it started.
- **Power lines**: coloured by what a fire starting there reaches. Click a line for figures.
- **Conservation land**: shaded darker where more of it is reachable. Click for figures.
- **Buildings**: coloured by how soon fire could arrive. Click for figures.

## Limitations

- **Wind speeds are probably low.** The weather record (ERA5) averages over about 25 km and smooths out valley and ridge winds. To test stronger wind, raise `weather.speed_scale` in `config.yaml` and rerun from step 3.
- **Not every spread model has been checked against the manual's tables.** The grass models and the slope factor are tested. The scrub model is a curve fitted to the manual's table, which its authors call interim. The conifer and indigenous forest coefficients come from the Canadian system and have not been compared with the manual.
- **Conifer forest is modelled as surface fire only.** Crown fire in pine and wilding conifers would spread faster.
- **Terrain is smoothed.** The elevation model is interpolated from 20 m contours.
- **Land cover is a snapshot** with a 1 ha minimum mapped area. Grazing, mowing and seasonal growth are not captured. Irrigated-land data dates from 2019-20.
- **Home versus business is approximate.** A building is counted as a business if it sits in an OpenStreetMap commercial, industrial or retail area. Sheds and outbuildings are included in building counts.
- **All power lines are treated as overhead**, and every point on them as equally likely to start a fire. The map shows consequence if a fire starts, not likelihood.
- **Fires starting outside the two districts are not simulated**, though fires can spread out into the surrounding area.
- **Spotting** (embers starting new fires ahead of the front) is not modelled, apart from letting fire cross streams and roads.
- Simulated fires come out about 10% smaller in area than the ideal ellipse because the grid allows only 16 directions of travel.

## Data sources and licences

| Data | Source | Licence |
|---|---|---|
| Elevation (NZ 8 m DEM) | Toitū Te Whenua LINZ | CC BY 4.0 |
| Land cover (LCDB v6.0) | Manaaki Whenua Landcare Research | CC BY 4.0 |
| Building outlines, Topo50 rivers, lakes, roads | Toitū Te Whenua LINZ | CC BY 4.0 |
| Public conservation land | Department of Conservation | CC BY 4.0 |
| Irrigated land (2020) | Ministry for the Environment | CC BY 4.0 |
| District boundaries, business land use | OpenStreetMap contributors | ODbL |
| LiDAR ground and surface models (1 m) | Toitū Te Whenua LINZ and regional councils | CC BY 4.0 |
| Sentinel-2 imagery | Copernicus / ESA, via Element 84 Earth Search | Free and open |
| Fire stations | OpenStreetMap contributors | ODbL |
| Weather (ERA5) | Copernicus Climate Change Service, via Open-Meteo | CC BY 4.0; Open-Meteo's free service is for non-commercial use |
| Power lines | `kml.kmz`, supplied | as supplied |

For commercial use, replace the Open-Meteo download with ERA5 from the Copernicus Climate Data Store.
