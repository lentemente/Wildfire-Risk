"""Directional rate-of-spread field for each wind scenario.

Outputs in data/work, per scenario: h_<id>.npy (head ROS, m/min),
a_<id>.npy, ex_<id>.npy, ey_<id>.npy (ellipse terms used by the simulator)
"""
from __future__ import annotations

import json

import numpy as np

import firelib as fl

CFG = fl.load_config()
BLOCK = 512


def main():
    sp = CFG["spread"]
    cell = CFG["cell_m"]
    fuels = fl.fuel_rows(CFG)
    # pasture curing varies cell by cell (curing.tif), so the table holds the
    # fully cured rate and the per-cell curing factor is applied afterwards
    tab = fl.ros_table([dict(f, curing_pct=100.0) if f["model"] == "o1a" else f for f in fuels])
    grass = fl.grass_rows(fuels)
    scen = json.loads((fl.WORK / "scenarios.json").read_text())

    dem = fl.read_tif("dem.tif")
    gy_s, gx = np.gradient(dem, cell)       # axis 0 runs south
    gy = -gy_s
    del dem, gy_s
    fuel = fl.read_tif("fuel_row.tif")
    breaks = fl.read_tif("breaks.tif")
    factor = np.ones(fuel.shape, dtype=np.float32)
    factor[(breaks & 1) > 0] *= sp["stream_factor"]
    factor[(breaks & 2) > 0] *= sp["road_factor"]
    del breaks
    curing = fl.read_tif("curing.tif")
    cf = np.clip(0.02 * curing.astype(np.float32) - 1.0, 0.0, 1.0)
    cf[curing == 255] = 1.0
    factor *= cf
    del curing, cf
    burnable = np.array([f["model"] != "none" and f["ros_mult"] > 0 for f in fuels])[fuel]

    for s in scen:
        wexp = fl.read_tif(f"wexp_{s['id']}.tif")
        h = np.zeros(fuel.shape, dtype=np.float32)
        a = np.zeros_like(h)
        ex = np.zeros_like(h)
        ey = np.zeros_like(h)
        to_deg = (s["from_deg"] + 180.0) % 360.0
        for r in range(0, fuel.shape[0], BLOCK):
            sl = slice(r, r + BLOCK)
            wind = s["speed_kmh"] * wexp[sl].astype(np.float64) / 100.0
            h[sl], a[sl], ex[sl], ey[sl] = fl.spread_field(
                fuel[sl], tab, grass, gx[sl], gy[sl], wind, to_deg, s["ffmc"],
                factor=factor[sl], max_slope_pct=sp["max_slope_pct"])
        for name, arr in (("h", h), ("a", a), ("ex", ex), ("ey", ey)):
            np.save(fl.WORK / f"{name}_{s['id']}.npy", arr)
        hb = h[burnable]
        q = np.percentile(hb[::50], [50, 90, 99])
        print(f"{s['id']}: head ROS on burnable ground median {q[0]:.1f}, "
              f"90th {q[1]:.1f}, 99th {q[2]:.1f} m/min "
              f"(= {q[0] * 0.06:.1f} / {q[1] * 0.06:.1f} / {q[2] * 0.06:.1f} km/h)")
        assert np.isfinite(h).all() and (a >= 0).all()
        assert (np.hypot(ex, ey) < 1).all()


if __name__ == "__main__":
    main()
