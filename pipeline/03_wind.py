"""Wind scenarios from the ERA5 record, and terrain wind exposure for each.

Outputs in data/work: scenarios.json, wexp_<id>.tif (wind multiplier x 100)
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

import firelib as fl

CFG = fl.load_config()
SECTORS = ["n", "ne", "e", "se", "s", "sw", "w", "nw"]
NAMES = ["Northerly", "North-easterly", "Easterly", "South-easterly",
         "Southerly", "South-westerly", "Westerly", "North-westerly"]


def scenarios() -> list[dict]:
    w = CFG["weather"]
    df = pd.read_csv(fl.RAW / "weather.csv.gz")
    df = df.dropna(subset=["temperature_2m", "relative_humidity_2m",
                           "wind_speed_10m", "wind_direction_10m"])
    hour = df["time"].str.slice(11, 13).astype(int)
    h0, h1 = w["afternoon_local_hours"]
    aft = df[(hour >= h0) & (hour <= h1)]
    hot = aft[(aft.temperature_2m >= w["min_temp_c"]) & (aft.relative_humidity_2m <= w["max_rh_pct"])].copy()
    print(f"afternoon hours: {len(aft)}, hot and dry: {len(hot)} ({len(hot) / len(aft):.1%})")
    hot["sector"] = (np.round(hot.wind_direction_10m / 45.0).astype(int)) % 8
    strong = hot[hot.wind_speed_10m >= hot.wind_speed_10m.quantile(0.75)]
    share = strong.sector.value_counts(normalize=True)
    print("strong-wind share by direction:",
          {SECTORS[s]: round(v, 2) for s, v in share.items()})
    out = []
    for s in share.index[: w["n_scenarios"]]:
        sec = hot[hot.sector == s]
        st = strong[strong.sector == s]
        temp, rh = float(st.temperature_2m.median()), float(st.relative_humidity_2m.median())
        out.append({
            "id": SECTORS[s], "name": NAMES[s], "from_deg": 45.0 * s,
            "speed_kmh": round(float(np.percentile(sec.wind_speed_10m, w["wind_percentile"]))
                               * float(w.get("speed_scale", 1.0)), 1),
            "temp_c": round(temp, 1), "rh_pct": round(rh, 1),
            "ffmc": round(fl.ffmc_from_weather(temp, rh), 1),
            "share": round(float(share[s]), 3)})
    for sc in out:
        sc["isi_open"] = round(float(fl.isi(sc["speed_kmh"], sc["ffmc"])), 1)
        print(" ", sc)
    return out


def exposure(dem: np.ndarray, from_deg: float, cell: float) -> np.ndarray:
    """Wind multiplier from the steepest angle up to terrain upwind."""
    we = CFG["wind_exposure"]
    th = math.radians(from_deg)
    ux, uy = math.sin(th), math.cos(th)          # unit vector pointing upwind
    nrow, ncol = dem.shape
    best = np.full(dem.shape, -np.inf, dtype=np.float32)
    for d in np.arange(we["step_m"], we["reach_m"] + 1, we["step_m"]):
        dc = int(round(ux * d / cell))
        dr = int(round(-uy * d / cell))
        if dr == 0 and dc == 0:
            continue
        dist = math.hypot(dr, dc) * cell
        # destination cell (r, c) looks at source cell (r + dr, c + dc)
        r0, r1 = max(0, -dr), min(nrow, nrow - dr)
        c0, c1 = max(0, -dc), min(ncol, ncol - dc)
        tan = (dem[r0 + dr:r1 + dr, c0 + dc:c1 + dc] - dem[r0:r1, c0:c1]) / dist
        np.maximum(best[r0:r1, c0:c1], tan, out=best[r0:r1, c0:c1])
    best[~np.isfinite(best)] = 0.0
    ang = np.degrees(np.arctan(best))
    mult = np.interp(ang, we["angle_deg"], we["multiplier"]).astype(np.float32)
    return uniform_filter(mult, size=5, mode="nearest")


def main():
    sc = scenarios()
    (fl.WORK / "scenarios.json").write_text(json.dumps(sc, indent=1))
    dem = fl.read_tif("dem.tif")
    cell = CFG["cell_m"]
    for s in sc:
        m = exposure(dem, s["from_deg"], cell)
        fl.write_tif(f"wexp_{s['id']}.tif", np.round(m * 100).astype(np.uint8))
        print(f"  exposure {s['id']}: mean x{m.mean():.2f}, "
              f"sheltered (<0.8) {(m < 0.8).mean():.0%}, exposed (>1.1) {(m > 1.1).mean():.0%}")


if __name__ == "__main__":
    main()
