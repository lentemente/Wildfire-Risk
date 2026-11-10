"""Simulate a fire from every ignition point on the power lines.

Outputs in data/work: ignitions.csv (one row per ignition point with tallies
per scenario), burncount_<id>.npy (fires reaching each cell), arrival_<id>.npy
(minutes until fire from anywhere on the network reaches each cell).

Usage: python pipeline/05_simulate.py [--limit N]   (N ignitions, for a trial)
"""
from __future__ import annotations

import json
import sys
import time

import geopandas as gpd
import numpy as np
import pandas as pd

import firelib as fl

CFG = fl.load_config()
BATCH = 480


def ignition_points(tr, shape, burnable):
    lines = gpd.read_file(fl.RAW / "lines.gpkg")
    spacing = CFG["simulation"]["ignition_spacing_m"]
    recs = []
    for lid, geom in zip(lines.line_id, lines.geometry):
        for part in getattr(geom, "geoms", [geom]):
            n = max(1, int(round(part.length / spacing)))
            for i in range(n):
                p = part.interpolate((i + 0.5) / n, normalized=True)
                recs.append((lid, p.x, p.y))
    df = pd.DataFrame(recs, columns=["line_id", "x", "y"])
    cell = CFG["cell_m"]
    col = np.floor((df.x.values - tr.c) / cell).astype(int)
    row = np.floor((tr.f - df.y.values) / cell).astype(int)
    seed = np.full(len(df), -1, dtype=np.int64)
    todo = (row >= 0) & (row < shape[0]) & (col >= 0) & (col < shape[1])
    # start on the nearest burnable cell within two cells of the line
    offs = sorted(((a, b) for a in range(-2, 3) for b in range(-2, 3)),
                  key=lambda o: o[0] ** 2 + o[1] ** 2)
    for dr, dc in offs:
        rr, cc = row + dr, col + dc
        ok = todo & (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
        ok[ok] = burnable[rr[ok], cc[ok]]
        seed[ok] = rr[ok] * shape[1] + cc[ok]
        todo &= ~ok
    df["seed"] = seed
    return df


def main():
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    sim = CFG["simulation"]
    cell = float(CFG["cell_m"])
    tmax = float(sim["burn_hours"]) * 60.0
    nthreads = int(sim["threads"])
    import numba
    numba.set_num_threads(min(nthreads, numba.config.NUMBA_NUM_THREADS))
    g, tr = fl.grid_profile()
    shape = (g["nrow"], g["ncol"])
    scen = json.loads((fl.WORK / "scenarios.json").read_text())
    dr, dc = fl.neighbours()

    pcl = (fl.read_tif("pcl.tif") > 0).astype(np.uint8)
    bld = fl.read_tif("bld.tif")
    bldc = fl.read_tif("bldc.tif")

    ign = None
    ha = cell * cell / 1e4
    for s in scen:
        sid = s["id"]
        a = np.load(fl.WORK / f"a_{sid}.npy")
        ex = np.load(fl.WORK / f"ex_{sid}.npy")
        ey = np.load(fl.WORK / f"ey_{sid}.npy")
        if ign is None:
            ign = ignition_points(tr, shape, a > 0)
            if limit:
                ign = ign.sample(n=min(limit, len(ign)), random_state=1).sort_index().reset_index(drop=True)
            print(f"{len(ign)} ignition points, {(ign.seed < 0).sum()} on non-burnable ground")
            resp = fl.read_tif("response.tif").ravel()
            ign["resp_min"] = np.where(ign.seed >= 0, resp[np.maximum(ign.seed.values, 0)], -1).round(0)
            del resp
        seeds = ign.seed.values.astype(np.int64)
        tresp = ign.resp_min.values.astype(np.float32)

        tstack = np.full((nthreads,) + shape, np.inf, dtype=np.float32)
        counts = np.zeros((nthreads,) + shape, dtype=np.uint16)
        out = np.zeros((len(seeds), 7), dtype=np.float32)
        t0 = time.time()
        for i in range(0, len(seeds), BATCH):
            out[i:i + BATCH] = fl.simulate(seeds[i:i + BATCH], tresp[i:i + BATCH],
                                           a, ex, ey, cell, tmax,
                                           pcl, bld, bldc, tstack, counts, dr, dc)
            done = min(i + BATCH, len(seeds))
            el = time.time() - t0
            print(f"  {sid}: {done}/{len(seeds)} fires, {el / 60:.1f} min elapsed, "
                  f"about {el / done * (len(seeds) - done) / 60:.1f} min left", flush=True)
        total = counts.sum(axis=0, dtype=np.uint32)
        del counts
        np.save(fl.WORK / f"burncount_{sid}.npy", np.minimum(total, 65535).astype(np.uint16))

        # arrival time from the network as a whole
        t = tstack[0]
        fl.burn(seeds[seeds >= 0], a, ex, ey, cell, tmax, t, dr, dc)
        np.save(fl.WORK / f"arrival_{sid}.npy", t)
        del tstack

        ign[f"ha_{sid}"] = (out[:, 0] * ha).round(0)
        ign[f"pcl_ha_{sid}"] = (out[:, 1] * ha).round(0)
        ign[f"bld_{sid}"] = out[:, 2].astype(int)
        ign[f"bldc_{sid}"] = out[:, 3].astype(int)
        ign[f"tfirst_{sid}"] = np.where(np.isfinite(out[:, 4]), out[:, 4], -1).round(0)
        ign[f"ha_arr_{sid}"] = (out[:, 5] * ha).round(1)
        ign[f"bld_arr_{sid}"] = out[:, 6].astype(int)
        v = ign[f"ha_{sid}"]
        print(f"{sid}: burned area per fire (ha) median {v.median():.0f}, "
              f"90th {v.quantile(0.9):.0f}, max {v.max():.0f}; "
              f"fires reaching a building {(ign[f'bld_{sid}'] > 0).mean():.0%}, "
              f"reaching conservation land {(ign[f'pcl_ha_{sid}'] > 0).mean():.0%}")
    ign.to_csv(fl.WORK / ("ignitions_trial.csv" if limit else "ignitions.csv"), index=False)


if __name__ == "__main__":
    main()
