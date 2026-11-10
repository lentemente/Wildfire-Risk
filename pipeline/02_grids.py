"""Build the analysis rasters: terrain, fuel, barriers and assets.

Outputs in data/work: grid.json, dem.tif, fuel_row.tif, breaks.tif, pcl.tif,
bld.tif, bldc.tif, buildings.csv
"""
from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features

import firelib as fl

CFG = fl.load_config()
RAW, WORK = fl.RAW, fl.WORK
WORK.mkdir(parents=True, exist_ok=True)


def need(name):
    p = RAW / f"{name}.gpkg"
    if not p.exists():
        raise SystemExit(f"Missing {p.name}. Run pipeline/01_fetch.py first "
                         "(the LINZ and LRIS layers need API keys in .env).")
    return gpd.read_file(p)


def burn_shapes(geoms, shape, tr, value=1, dtype="uint8", all_touched=False, out=None):
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if out is None:
        out = np.zeros(shape, dtype=dtype)
    if geoms:
        features.rasterize(((g, value) for g in geoms), out=out, transform=tr,
                           all_touched=all_touched)
    return out


def main():
    with rasterio.open(RAW / "dem.tif") as src:
        dem = src.read(1)
        tr = src.transform
    shape = dem.shape
    cell = CFG["cell_m"]
    (WORK / "grid.json").write_text(json.dumps({
        "transform": list(tr)[:6], "nrow": shape[0], "ncol": shape[1], "cell": cell}))
    # LiDAR ground model replaces the contour-derived DEM wherever it was flown
    canopy = None
    if (RAW / "lidar_dem.tif").exists():
        with rasterio.open(RAW / "lidar_dem.tif") as src:
            lid = src.read(1)
        has = np.isfinite(lid)
        diff = np.abs(lid[has][::101] - dem[has][::101])
        print(f"LiDAR terrain covers {has.mean():.1%} of the grid; typical difference "
              f"from the 8 m DEM {np.nanmedian(diff):.1f} m")
        dem[has] = lid[has]
        if (RAW / "lidar_dsm.tif").exists():
            with rasterio.open(RAW / "lidar_dsm.tif") as src:
                canopy = np.clip(src.read(1) - lid, 0, None).astype(np.float32)
        del lid, has
    missing = np.isnan(dem)
    print(f"grid {shape[1]} x {shape[0]} cells; DEM missing {missing.mean():.2%}")
    dem[missing] = 0.0
    fl.write_tif("dem.tif", dem)
    del dem

    # ---- fuel ------------------------------------------------------------
    fuels = fl.fuel_rows(CFG)
    row_of = {f["lcdb_class"]: i for i, f in enumerate(fuels)}
    none_row = row_of[0]
    lcdb = need("lcdb")
    cls_col = next((c for c in ("Class_2023", "Class_2018") if c in lcdb.columns), None)
    if cls_col is None:
        raise SystemExit(f"No LCDB class column found; columns are {list(lcdb.columns)}")
    name_col = cls_col.replace("Class", "Name")
    lcdb[cls_col] = lcdb[cls_col].astype(int)
    unknown = sorted(set(lcdb[cls_col]) - set(row_of))
    if unknown:
        names = lcdb[lcdb[cls_col].isin(unknown)].groupby(cls_col)[name_col].first().to_dict()
        print("WARNING: LCDB classes missing from fuels.csv, treated as non-fuel:", names)
    lcdb["row"] = lcdb[cls_col].map(row_of).fillna(none_row).astype(int)
    fuel = np.full(shape, none_row, dtype=np.uint8)
    for r, grp in lcdb.groupby("row"):
        burn_shapes(grp.geometry, shape, tr, value=int(r), out=fuel)
    del lcdb

    irr_path = RAW / "irrigated.gpkg"
    if irr_path.exists():
        irr = burn_shapes(gpd.read_file(irr_path).geometry, shape, tr)
        grass = np.isin(fuel, [row_of[c] for c in (30, 40, 41)])
        fuel[(irr == 1) & grass] = len(fuels) - 1
        print(f"irrigated pasture: {((irr == 1) & grass).sum() * cell * cell / 1e6:.0f} km2")
        del irr, grass
    else:
        print("no irrigated-land layer: irrigated pasture treated as ordinary pasture")

    # open water always blocks, even where LCDB maps the banks as vegetation
    water_row = row_of[20]
    for name in ("lake_poly", "river_poly"):
        burn_shapes(need(name).geometry, shape, tr, value=water_row, out=fuel,
                    all_touched=True)
    km2 = cell * cell / 1e6
    b_raw = need("buildings")

    # ---- LiDAR vegetation height refines grass and tussock ---------------------
    if canopy is not None:
        lc = CFG["lidar"]
        roofs = burn_shapes(b_raw.geometry, shape, tr, all_touched=True) == 1
        open_rows = [i for i, f in enumerate(fuels) if f["fuel"] in ("grass", "tussock", "irrigated")]
        is_open = np.isin(fuel, open_rows) & ~roofs & np.isfinite(canopy)
        to_tree = is_open & (canopy >= lc["tree_min_m"])
        to_scrub = is_open & (canopy >= lc["scrub_min_m"]) & ~to_tree
        fuel[to_scrub] = row_of[56]          # mixed exotic shrubland -> scrub model
        fuel[to_tree] = row_of[71]           # exotic forest -> conifer model
        print(f"LiDAR vegetation height: {is_open.sum() * km2:.0f} km2 of grass/tussock checked; "
              f"{to_scrub.sum() * km2:.0f} km2 reclassed to scrub, {to_tree.sum() * km2:.0f} km2 to trees")
        canopy[roofs] = np.nan
        fl.write_tif("canopy.tif", canopy, nodata=np.nan)
        del roofs, is_open, to_tree, to_scrub, canopy
    fl.write_tif("fuel_row.tif", fuel)

    # ---- satellite greenness sets grass curing ---------------------------------
    curing = np.array([f["curing_pct"] for f in fuels], dtype=np.float32)[fuel]
    is_o1a = np.array([f["model"] == "o1a" for f in fuels])[fuel]
    if (RAW / "ndvi.tif").exists():
        sat = CFG["satellite"]
        with rasterio.open(RAW / "ndvi.tif") as src:
            ndvi = src.read(1)
        use = is_o1a & np.isfinite(ndvi)
        curing[use] = np.interp(ndvi[use], [sat["ndvi_cured"], sat["ndvi_green"]], [100.0, 50.0])
        by = {}
        for name in ("grass", "irrigated", "crop"):
            m = use & np.isin(fuel, [i for i, f in enumerate(fuels) if f["fuel"] == name])
            if m.any():
                by[name] = (round(float(np.median(ndvi[m][::20])), 2), round(float(np.median(curing[m][::20]))))
        print(f"satellite curing applied to {use.sum() * km2:.0f} km2 of pasture and crops; "
              f"median (NDVI, curing %) by type: {by}")
        del ndvi, use
    too_green = is_o1a & (curing <= 50.0)        # will not carry fire
    curing[~is_o1a] = 255
    fl.write_tif("curing.tif", np.round(curing).astype(np.uint8))
    del curing, is_o1a

    counts = np.bincount(fuel.ravel(), minlength=len(fuels))
    by_fuel = {}
    for i, f in enumerate(fuels):
        by_fuel[f["fuel"]] = by_fuel.get(f["fuel"], 0) + counts[i] * km2
    print("area by fuel type (km2):", {k: round(v) for k, v in sorted(by_fuel.items(), key=lambda kv: -kv[1])})
    burnable = np.array([f["model"] != "none" and f["ros_mult"] > 0 for f in fuels])[fuel] & ~too_green
    print(f"pasture too green to burn: {too_green.sum() * km2:.0f} km2")
    del too_green

    # ---- partial barriers ---------------------------------------------------
    breaks = np.zeros(shape, dtype=np.uint8)
    streams = need("river_line")
    breaks |= burn_shapes(streams.geometry, shape, tr, value=1)
    roads = need("roads")
    if "surface" in roads.columns:
        roads = roads[roads["surface"].astype(str).str.lower() == "sealed"]
    breaks |= burn_shapes(roads.geometry, shape, tr, value=2)
    fl.write_tif("breaks.tif", breaks)
    print(f"stream cells {(breaks & 1).mean():.1%}, sealed-road cells {(breaks & 2 > 0).mean():.1%}")
    del breaks, streams, roads

    # ---- conservation land ----------------------------------------------------
    pcl = gpd.read_file(RAW / "pcl.gpkg")
    pcl["pcl_id"] = np.arange(1, len(pcl) + 1)
    pcl_r = np.zeros(shape, dtype=np.uint16)
    features.rasterize(((g, i) for g, i in zip(pcl.geometry, pcl.pcl_id)
                        if g is not None and not g.is_empty),
                       out=pcl_r, transform=tr)
    fl.write_tif("pcl.tif", pcl_r)
    print(f"conservation land: {len(pcl)} areas, {(pcl_r > 0).sum() * km2:.0f} km2 in the grid")
    del pcl_r

    # ---- buildings --------------------------------------------------------------
    b = b_raw
    b = b[b.geometry.notna() & ~b.geometry.is_empty].reset_index(drop=True)
    cen = b.geometry.centroid
    biz = gpd.read_file(RAW / "osm_business.gpkg")
    pts = gpd.GeoDataFrame({"i": np.arange(len(b))}, geometry=cen, crs=b.crs)
    hit = gpd.sjoin(pts, biz[["geometry"]], predicate="within", how="inner")["i"].unique()
    commercial = np.zeros(len(b), dtype=bool)
    commercial[hit] = True
    col = np.floor((cen.x.values - tr.c) / cell).astype(int)
    row = np.floor((tr.f - cen.y.values) / cell).astype(int)
    inside = (row >= 0) & (row < shape[0]) & (col >= 0) & (col < shape[1])

    # a building on non-burnable ground counts as exposed if burnable ground
    # lies within building_reach_m; it is tallied at that nearest burnable cell
    reach = int(round(CFG["simulation"]["building_reach_m"] / cell))
    offs = sorted(((dr, dc) for dr in range(-reach, reach + 1) for dc in range(-reach, reach + 1)),
                  key=lambda o: o[0] ** 2 + o[1] ** 2)
    arow = np.full(len(b), -1)
    acol = np.full(len(b), -1)
    todo = inside.copy()
    for dr, dc in offs:
        rr, cc = row + dr, col + dc
        ok = todo & (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
        ok[ok] = burnable[rr[ok], cc[ok]]
        arow[ok], acol[ok] = rr[ok], cc[ok]
        todo &= ~ok
    exposed = arow >= 0
    bld = np.zeros(shape, dtype=np.uint16)
    bldc = np.zeros(shape, dtype=np.uint16)
    np.add.at(bld, (arow[exposed], acol[exposed]), 1)
    np.add.at(bldc, (arow[exposed & commercial], acol[exposed & commercial]), 1)
    fl.write_tif("bld.tif", bld)
    fl.write_tif("bldc.tif", bldc)
    use = b["use"].fillna("Unknown").astype(str) if "use" in b.columns else "Unknown"
    out = pd.DataFrame({
        "x": cen.x.round(1), "y": cen.y.round(1), "area_m2": b.geometry.area.round(0),
        "use": use, "commercial": commercial.astype(int), "row": arow, "col": acol})
    out[inside].to_csv(WORK / "buildings.csv", index=False)
    print(f"buildings: {inside.sum()} in the grid, {exposed.sum()} within "
          f"{CFG['simulation']['building_reach_m']} m of burnable ground, "
          f"{commercial[inside].sum()} in commercial/industrial land use")


if __name__ == "__main__":
    main()
