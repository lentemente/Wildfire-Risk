"""Publish results for the web viewer: map tiles, GeoJSON and legends.

Outputs: web/tiles/<layer>/<z>/<x>/<y>.png, web/data/*.geojson,
web/data/meta.json, web/config.js

Usage: python pipeline/06_export.py [base] [rasters] [vectors]   (default: all)
"""
from __future__ import annotations

import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import mercantile
import numpy as np
import pandas as pd
from PIL import Image
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject

import firelib as fl

CFG = fl.load_config()
ZMIN, ZMAX = CFG["export"]["min_zoom"], CFG["export"]["max_zoom"]
TILES = fl.WEB / "tiles"
DATA = fl.WEB / "data"
T = 256

# Sequential ramps: one hue, light to dark. Categorical: fixed order.
RED = ["#fcd9cf", "#f5a48c", "#e86a4f", "#c93a2c", "#8a1616"]
VIOLET = ["#dcd6f7", "#b3a8ee", "#8a78e0", "#6250d6", "#3a2a8f"]
TIME = ["#8a1616", "#c93a2c", "#e86a4f", "#f5a48c"]            # soonest = darkest
DIVERGE = ["#256abf", "#86b6ef", None, "#f0a070", "#c9481e"]   # sheltered .. exposed
FUEL_COLOURS = {
    "grass": "#eda100", "tussock": "#eb6834", "scrub": "#e34948",
    "hardwood": "#e87ba4", "pine": "#008300", "indigenous": "#1baf7a",
    "deciduous": "#6250d6", "crop": "#2a78d6", "irrigated": "#9ec5f4"}
FUEL_LABELS = {
    "grass": "Pasture and short grass", "tussock": "Tussock and rank grass",
    "scrub": "Scrub (manuka, gorse, matagouri)", "hardwood": "Hardwood and sub-alpine shrubs",
    "pine": "Exotic conifer forest", "indigenous": "Indigenous forest",
    "deciduous": "Deciduous trees", "crop": "Crops, orchards, vineyards",
    "irrigated": "Irrigated pasture"}

GREEN = ["#cfe8c9", "#93cd8a", "#4fa84a", "#1f7a1f", "#0b4d0b"]
BROWN = ["#f3e3c3", "#e3c184", "#cf9a45", "#a86f1c", "#6e4409"]

LEGENDS = {
    "canopy": {"title": "Vegetation height from LiDAR", "unit": "m",
               "breaks": [0.3, 1, 2, 5, 10], "colours": [None] + GREEN,
               "labels": ["under 0.3 (not shaded)", "0.3 to 1", "1 to 2", "2 to 5", "5 to 10", "over 10"],
               "note": "Blank also where no LiDAR has been flown."},
    "curing": {"title": "Pasture curing from satellite", "unit": "% dead grass",
               "breaks": [50.5, 60, 70, 80, 90], "colours": [None] + BROWN,
               "labels": ["50 or less: too green to burn (not shaded)", "51 to 60", "60 to 70",
                          "70 to 80", "80 to 90", "90 to 100"],
               "note": "Pasture and crops only."},
    "ros": {"title": "Head-fire rate of spread", "unit": "m/min",
            "breaks": [2, 10, 25, 50], "colours": RED,
            "labels": ["under 2", "2 to 10", "10 to 25", "25 to 50", "over 50"]},
    "intensity": {"title": "Head-fire intensity", "unit": "kW/m",
                  "breaks": [500, 2000, 4000, 10000], "colours": RED,
                  "labels": ["under 500", "500 to 2,000", "2,000 to 4,000",
                             "4,000 to 10,000", "over 10,000"]},
    "arrival": {"title": "Time for fire to arrive from the nearest line", "unit": "hours",
                "breaks": [60, 120, 240], "colours": TIME,
                "labels": ["under 1", "1 to 2", "2 to 4", "4 to 6"]},
    "wexp": {"title": "Wind exposure", "unit": "x open wind",
             "breaks": [70, 90, 105, 120], "colours": DIVERGE,
             "labels": ["under 0.7 (sheltered)", "0.7 to 0.9", "0.9 to 1.05",
                        "1.05 to 1.2", "over 1.2 (exposed)"]},
}


def rgba(hex_or_none, alpha=255):
    if hex_or_none is None:
        return (0, 0, 0, 0)
    h = hex_or_none.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha)


# ---------------------------------------------------------------------------
class Mosaic:
    """A web-mercator image aligned to the tile grid at one zoom level."""

    def __init__(self, arr, z, x0, y0):
        self.arr, self.z, self.x0, self.y0 = arr, z, x0, y0

    @classmethod
    def warp(cls, src, tr, resampling, nodata=np.nan):
        g, _ = fl.grid_profile()
        w, n = tr * (0, 0)
        e, s = tr * (g["ncol"], g["nrow"])
        ll = gpd.GeoSeries(gpd.points_from_xy([w, e, w, e], [s, s, n, n]), crs=2193).to_crs(4326)
        ul = mercantile.tile(ll.x.min(), ll.y.max(), ZMAX)
        lr = mercantile.tile(ll.x.max(), ll.y.min(), ZMAX)
        nx, ny = lr.x - ul.x + 1, lr.y - ul.y + 1
        b0, b1 = mercantile.xy_bounds(ul), mercantile.xy_bounds(lr)
        dtr = from_bounds(b0.left, b1.bottom, b1.right, b0.top, nx * T, ny * T)
        dst = np.full((ny * T, nx * T), nodata, dtype=src.dtype)
        reproject(src, dst, src_transform=tr, src_crs="EPSG:2193", src_nodata=nodata,
                  dst_transform=dtr, dst_crs="EPSG:3857", dst_nodata=nodata,
                  resampling=resampling, num_threads=8)
        return cls(dst, ZMAX, ul.x, ul.y)

    def parent(self, categorical):
        a, x0, y0 = self.arr, self.x0, self.y0
        fill = a.dtype.type(255) if categorical else np.nan
        padl, padt = (x0 % 2) * T, (y0 % 2) * T
        padr = ((x0 + a.shape[1] // T) % 2) * T
        padb = ((y0 + a.shape[0] // T) % 2) * T
        a = np.pad(a, ((padt, padb), (padl, padr)), constant_values=fill)
        if categorical:
            a = a[::2, ::2]
        else:
            v = a.reshape(a.shape[0] // 2, 2, a.shape[1] // 2, 2)
            ok = np.isfinite(v)
            n = ok.sum(axis=(1, 3))
            a = np.where(n > 0, np.where(ok, v, 0).sum(axis=(1, 3)) / np.maximum(n, 1), np.nan).astype(np.float32)
        return Mosaic(a, self.z - 1, x0 // 2, y0 // 2)

    def tiles(self):
        a = self.arr
        for j in range(a.shape[0] // T):
            for i in range(a.shape[1] // T):
                yield self.x0 + i, self.y0 + j, a[j * T:(j + 1) * T, i * T:(i + 1) * T]


def write_layer(name, src, tr, colourise, categorical=False, resampling=None, keep_empty=False):
    out = TILES / name
    if out.exists():
        shutil.rmtree(out)
    res = resampling or (Resampling.nearest if categorical else Resampling.bilinear)
    m = Mosaic.warp(src, tr, res, nodata=255 if categorical else np.nan)
    n = 0
    with ThreadPoolExecutor(8) as pool:
        while True:
            jobs = []
            for x, y, block in m.tiles():
                empty = (block == 255).all() if categorical else not np.isfinite(block).any()
                if empty and not keep_empty:
                    continue
                img = colourise(block)
                d = out / str(m.z) / str(x)
                d.mkdir(parents=True, exist_ok=True)
                jobs.append(pool.submit(Image.fromarray(img).save, d / f"{y}.png"))
            for j in jobs:
                j.result()
            n += len(jobs)
            if m.z == ZMIN:
                break
            m = m.parent(categorical)
    print(f"  {name}: {n} tiles")


def classed(breaks, colours, alpha=215):
    lut = np.array([rgba(c, alpha) for c in colours] + [(0, 0, 0, 0)], dtype=np.uint8)

    def f(block):
        idx = np.digitize(block, breaks)
        idx[~np.isfinite(block)] = len(colours)
        return lut[idx]
    return f


def terrarium(block):
    v = np.where(np.isfinite(block), block, 0.0).astype(np.float64) + 32768.0
    out = np.empty(block.shape + (3,), dtype=np.uint8)
    out[..., 0] = np.floor(v / 256.0)
    out[..., 1] = np.floor(v) % 256
    out[..., 2] = np.floor((v - np.floor(v)) * 256.0)
    return out


def grey(block):
    ok = np.isfinite(block)
    v = np.where(ok, block, 0).astype(np.uint8)
    return np.dstack([v, v, v, np.where(ok, 255, 0).astype(np.uint8)])


# ---------------------------------------------------------------------------
def export_base(tr):
    dem = fl.read_tif("dem.tif")
    write_layer("terrain", dem, tr, terrarium, keep_empty=True)
    cell = CFG["cell_m"]
    gy, gx = np.gradient(dem, cell)
    del dem
    az, alt = np.radians(315.0), np.radians(45.0)
    slope = np.arctan(2.0 * np.hypot(gx, gy))          # 2x vertical exaggeration
    aspect = np.arctan2(gy, -gx)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - np.pi / 2 - aspect)
    hs = (150 + 105 * np.clip(hs, 0, 1)).astype(np.float32)   # light base: 150..255
    write_layer("hillshade", hs, tr, grey)


def nan0(a):
    a = a.astype(np.float32)
    a[a <= 0] = np.nan
    return a


def export_rasters(tr, scen, fuels, meta):
    fuel = fl.read_tif("fuel_row.tif")
    names = [k for k in FUEL_COLOURS]
    lut_idx = np.array([names.index(f["fuel"]) if f["fuel"] in names else 255 for f in fuels], dtype=np.uint8)
    lut = np.zeros((256, 4), dtype=np.uint8)
    for i, k in enumerate(names):
        lut[i] = rgba(FUEL_COLOURS[k], 200)
    write_layer("fuel", lut_idx[fuel], tr, lambda b: lut[b], categorical=True)
    load = np.array([f["load_t_ha"] for f in fuels], dtype=np.float32)[fuel]
    del fuel
    for key, fname in (("canopy", "canopy.tif"), ("curing", "curing.tif")):
        if not (fl.WORK / fname).exists():
            LEGENDS.pop(key, None)
            continue
        a = fl.read_tif(fname).astype(np.float32)
        if key == "curing":
            a[a == 255] = np.nan
        lg = LEGENDS[key]
        write_layer(key, a, tr, classed(lg["breaks"], lg["colours"]))
        del a

    worst = {}
    for s in scen:
        sid = s["id"]
        h = np.load(fl.WORK / f"h_{sid}.npy")
        bc = np.load(fl.WORK / f"burncount_{sid}.npy").astype(np.float32)
        arr = np.load(fl.WORK / f"arrival_{sid}.npy")
        arr[~np.isfinite(arr)] = np.nan
        worst["ros"] = h.copy() if "ros" not in worst else np.maximum(worst["ros"], h)
        worst["burn"] = bc.copy() if "burn" not in worst else np.maximum(worst["burn"], bc)
        worst["arrival"] = arr.copy() if "arrival" not in worst else np.fmin(worst["arrival"], arr)
        s["_layers"] = {"ros": h, "burn": bc, "arrival": arr}

    nz = worst["burn"][worst["burn"] > 0]
    q = np.percentile(nz[::20], [50, 75, 90, 97]) if nz.size else np.array([1, 2, 3, 4])
    breaks = sorted({max(2, int(round(v))) for v in q})
    while len(breaks) < 4:
        breaks.append(breaks[-1] * 2)
    spacing_km = CFG["simulation"]["ignition_spacing_m"] / 1000
    LEGENDS["burn"] = {
        "title": "Simulated line fires reaching this spot", "unit": "fires",
        "breaks": breaks, "colours": VIOLET,
        "labels": [f"1 to {breaks[0] - 1}"] + [f"{a} to {b - 1}" for a, b in zip(breaks, breaks[1:])]
                  + [f"{breaks[-1]} or more"],
        "note": f"One fire is simulated for about every {spacing_km:g} km of line."}

    def emit(sid, layers):
        for key, a in layers.items():
            lg = LEGENDS[key]
            write_layer(f"{key}_{sid}", nan0(a) if key != "arrival" else a, tr,
                        classed(lg["breaks"], lg["colours"]))
        lg = LEGENDS["intensity"]
        write_layer(f"intensity_{sid}", nan0(30.0 * load * layers["ros"]), tr,
                    classed(lg["breaks"], lg["colours"]))

    for s in scen:
        emit(s["id"], s.pop("_layers"))
        lg = LEGENDS["wexp"]
        write_layer(f"wexp_{s['id']}", fl.read_tif(f"wexp_{s['id']}.tif").astype(np.float32), tr,
                    classed(lg["breaks"], lg["colours"], alpha=190))
    emit("worst", worst)
    meta["legends"] = LEGENDS
    meta["fuel_legend"] = [{"label": FUEL_LABELS[k], "colour": FUEL_COLOURS[k]} for k in names]
    return worst


def export_vectors(tr, scen, meta):
    g, _ = fl.grid_profile()
    sids = [s["id"] for s in scen]
    DATA.mkdir(parents=True, exist_ok=True)

    def save(gdf, name, tol):
        gdf = gdf.copy()
        if tol:
            gdf["geometry"] = gdf.geometry.simplify(tol)
        gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].to_crs(4326)
        gdf.to_file(DATA / name, driver="GeoJSON", COORDINATE_PRECISION=5)
        print(f"  {name}: {len(gdf)} features, {(DATA / name).stat().st_size / 1e6:.1f} MB")

    save(gpd.read_file(fl.RAW / "districts.gpkg"), "districts.geojson", 50)

    # ---- power lines with the consequence of a fire starting on them --------
    ign = pd.read_csv(fl.WORK / "ignitions.csv")
    for k in ("ha", "pcl_ha", "bld", "bldc"):
        ign[f"{k}_worst"] = ign[[f"{k}_{s}" for s in sids]].max(axis=1)
    tf = ign[[f"tfirst_{s}" for s in sids]].where(lambda d: d >= 0)
    ign["tfirst_worst"] = tf.min(axis=1).fillna(-1)
    agg = {}
    for s in sids + ["worst"]:
        for k in ("ha", "pcl_ha", "bld", "bldc"):
            agg[f"{k}_{s}"] = "max"
        agg[f"tfirst_{s}"] = lambda v: v[v >= 0].min() if (v >= 0).any() else -1
    per_line = ign.groupby("line_id").agg(agg).reset_index()
    lines = gpd.read_file(fl.RAW / "lines.gpkg").merge(per_line, on="line_id", how="left").fillna(0)
    pr = pd.read_csv(fl.WORK / "priority.csv")
    band_names = list(CFG["priority"]["bands"]) + ["Low"]
    pr["band_i"] = pr["band"].map({n: i for i, n in enumerate(band_names)})
    lines = lines.merge(pr[["line_id", "rank", "score", "band", "band_i", "s_property",
                            "s_conservation", "s_response", "s_spread", "resp_min",
                            "ha_arr", "bld_arr"]], on="line_id", how="left")
    save(lines, "lines.geojson", 5)
    meta["priority"] = {
        "bands": band_names, "weights": CFG["priority"]["weights"],
        "commercial_multiplier": CFG["priority"]["commercial_multiplier"],
        "km": {n: round(float(lines.length_m[lines.band == n].sum() / 1000)) for n in band_names},
        "min_score": {n: float(lines.score[lines.band == n].min()) for n in band_names}}
    st = gpd.read_file(fl.RAW / "stations.gpkg")
    save(st, "stations.geojson", 0)
    meta["line_breaks"] = {}
    for k in ("ha", "pcl_ha", "bld"):
        v = lines[f"{k}_worst"]
        v = v[v > 0]
        qs = np.percentile(v, [50, 75, 90, 97]) if len(v) else [1, 2, 3, 4]
        br = sorted({float(f"{x:.2g}") for x in qs if x > 0})
        meta["line_breaks"][k] = br
    meta["summary"] = {
        "ignitions": int(len(ign)), "line_km": round(float(lines.length_m.sum() / 1000)),
        "lines": int(len(lines)),
        "by_scenario": {s: {
            "median_ha": float(ign[f"ha_{s}"].median()),
            "p90_ha": float(ign[f"ha_{s}"].quantile(0.9)),
            "share_reaching_building": round(float((ign[f"bld_{s}"] > 0).mean()), 3),
            "share_reaching_pcl": round(float((ign[f"pcl_ha_{s}"] > 0).mean()), 3)} for s in sids}}

    # ---- conservation land -----------------------------------------------------
    pcl_r = fl.read_tif("pcl.tif").ravel()
    pcl = gpd.read_file(fl.RAW / "pcl.gpkg")
    n = len(pcl) + 1
    cells = np.bincount(pcl_r, minlength=n)[:n]
    ha = CFG["cell_m"] ** 2 / 1e4
    cols = {"name": pcl["Name"].fillna(""), "type": pcl["Type"].fillna(""),
            "ha": (cells[1:] * ha).round(0)}
    arrivals = {s: np.load(fl.WORK / f"arrival_{s}.npy") for s in sids}
    arrivals["worst"] = np.fmin.reduce([arrivals[s] for s in sids])
    for s, arr in arrivals.items():
        a = arr.ravel()
        hit = np.isfinite(a)
        reached = np.bincount(pcl_r[hit], minlength=n)[:n]
        tmin = np.full(n, np.inf)
        np.minimum.at(tmin, pcl_r[hit], a[hit])
        cols[f"pct_{s}"] = np.where(cells[1:] > 0, 100.0 * reached[1:] / np.maximum(cells[1:], 1), 0).round(0)
        cols[f"t_{s}"] = np.where(np.isfinite(tmin[1:]), tmin[1:], -1).round(0)
    out = gpd.GeoDataFrame(cols, geometry=pcl.geometry.values, crs=pcl.crs)
    out = out[out.ha > 0]
    save(out, "pcl.geojson", 25)
    meta["summary"]["pcl_ha"] = float(out.ha.sum())
    meta["summary"]["pcl_ha_reached_worst"] = float((out.ha * out.pct_worst / 100).sum())

    # ---- buildings -------------------------------------------------------------
    b = pd.read_csv(fl.WORK / "buildings.csv")
    exposed = b.row.values >= 0
    idx = (b.row.values * g["ncol"] + b.col.values)[exposed]
    props = {"c": b.commercial.values, "m2": b.area_m2.values.astype(int)}
    burn = {s: np.load(fl.WORK / f"burncount_{s}.npy") for s in sids}
    burn["worst"] = np.maximum.reduce([burn[s] for s in sids])
    for s in sids + ["worst"]:
        nfire = np.zeros(len(b), dtype=int)
        nfire[exposed] = burn[s].ravel()[idx]
        t = np.full(len(b), -1.0)
        tv = arrivals[s].ravel()[idx]
        t[exposed] = np.where(np.isfinite(tv), tv, -1)
        props[f"n_{s}"] = nfire
        props[f"t_{s}"] = t.round(0)
    bg = gpd.GeoDataFrame(props, geometry=gpd.points_from_xy(b.x, b.y), crs=2193)
    save(bg, "buildings.geojson", 0)
    meta["summary"]["buildings"] = int(len(b))
    meta["summary"]["buildings_reached_worst"] = int((bg.t_worst >= 0).sum())
    meta["summary"]["business_buildings_reached_worst"] = int(((bg.t_worst >= 0) & (bg.c == 1)).sum())


def main():
    steps = sys.argv[1:] or ["base", "rasters", "vectors"]
    g, tr = fl.grid_profile()
    scen = json.loads((fl.WORK / "scenarios.json").read_text())
    fuels = fl.fuel_rows(CFG)
    DATA.mkdir(parents=True, exist_ok=True)
    mp = DATA / "meta.json"
    meta = json.loads(mp.read_text()) if mp.exists() else {}
    w, n = tr * (0, 0)
    e, s = tr * (g["ncol"], g["nrow"])
    ll = gpd.GeoSeries(gpd.points_from_xy([w, e, w, e], [s, s, n, n]), crs=2193).to_crs(4326)
    meta.update({
        "bounds": [float(ll.x.min()), float(ll.y.min()), float(ll.x.max()), float(ll.y.max())],
        "minzoom": ZMIN, "maxzoom": ZMAX, "burn_hours": CFG["simulation"]["burn_hours"],
        "scenarios": [{k: v for k, v in sc.items() if not k.startswith("_")} for sc in scen]})
    if "base" in steps:
        export_base(tr)
    if "rasters" in steps:
        export_rasters(tr, scen, fuels, meta)
    if "vectors" in steps:
        export_vectors(tr, scen, meta)
    mp.write_text(json.dumps(meta, indent=1))
    key = fl.load_env().get("LINZ_BASEMAPS_KEY", "")
    (fl.WEB / "config.js").write_text(f"window.LINZ_BASEMAPS_KEY = {json.dumps(key)};\n")
    print("export done")


if __name__ == "__main__":
    main()
