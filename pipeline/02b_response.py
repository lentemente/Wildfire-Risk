"""Estimate how soon the first fire crew could reach any point.

Travel is from the nearest fire station along roads (speed by surface), then
on foot or by 4WD across open ground. Water cannot be crossed except on a road.
Output: data/work/response.tif (minutes, including turnout time)
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
from rasterio import features

import firelib as fl

CFG = fl.load_config()


def main():
    rc = CFG["response"]
    g, tr = fl.grid_profile()
    shape = (g["nrow"], g["ncol"])
    cell = float(g["cell"])
    fuels = fl.fuel_rows(CFG)
    fuel = fl.read_tif("fuel_row.tif")
    water = np.isin(fuel, [i for i, f in enumerate(fuels) if f["lcdb_class"] in (20, 21, 22, 14)])
    del fuel

    kmh = rc["speed_kmh"]
    speed = np.full(shape, kmh["off_road"] * 1000 / 60, dtype=np.float32)   # m/min
    speed[water] = 0.0
    del water
    roads = gpd.read_file(fl.RAW / "roads.gpkg")
    surf = roads["surface"].astype(str).str.lower()
    for name in ("unmetalled", "metalled", "sealed"):          # fastest drawn last
        geoms = roads.geometry[surf == name]
        if len(geoms):
            features.rasterize(((gm, kmh[name] * 1000 / 60) for gm in geoms if gm is not None),
                               out=speed, transform=tr, all_touched=True)

    st = gpd.read_file(fl.RAW / "stations.gpkg")
    col = np.floor((st.geometry.x.values - tr.c) / cell).astype(int)
    row = np.floor((tr.f - st.geometry.y.values) / cell).astype(int)
    road_speed = kmh["unmetalled"] * 1000 / 60
    seeds = []
    for r, c in zip(row, col):
        speed[max(r - 2, 0):r + 3, max(c - 2, 0):c + 3] = np.maximum(
            speed[max(r - 2, 0):r + 3, max(c - 2, 0):c + 3], road_speed)   # station forecourt
        seeds.append(r * shape[1] + c)
    zero = np.zeros(shape, dtype=np.float32)
    t = np.full(shape, np.inf, dtype=np.float32)
    dr, dc = fl.neighbours()
    tmax = float(rc["max_min"])
    fl.burn(np.array(seeds, dtype=np.int64), speed, zero, zero, cell, tmax, t, dr, dc)
    t = np.where(np.isfinite(t), t, tmax) + float(rc["turnout_min"])
    fl.write_tif("response.tif", t.astype(np.float32))
    q = np.percentile(t[::7, ::7], [10, 50, 90])
    print(f"{len(st)} fire stations; crew arrival across the area: "
          f"10th {q[0]:.0f}, median {q[1]:.0f}, 90th {q[2]:.0f} min")


if __name__ == "__main__":
    main()
