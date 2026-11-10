"""Shared fire-behaviour equations, the travel-time kernel and small IO helpers.

Units: rate of spread (ROS) m/min, wind km/h, time minutes, distance metres.
Directions are (east, north) vectors; raster rows increase southwards.

Spread models are those named in Pearce & Anderson (2008), "A Manual for
Predicting Fire Behaviour in New Zealand Fuels" (Scion), which draws on the
Canadian FBP System (Forestry Canada 1992, ST-X-3) for grass and forest fuels.
"""
from __future__ import annotations

import csv
import math
import os
from pathlib import Path

import numpy as np
import yaml
from numba import njit, prange

ROOT = Path(__file__).resolve().parent.parent
# FIRE_DATA / FIRE_WEB redirect outputs, e.g. for a trial run on test data
_DATA = Path(os.environ.get("FIRE_DATA") or ROOT / "data")
RAW = _DATA / "raw"
WORK = _DATA / "work"
WEB = Path(os.environ.get("FIRE_WEB") or ROOT / "web")


# ----------------------------------------------------------------------------
# Config / env
# ----------------------------------------------------------------------------
def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_env() -> dict:
    """Read KEY=value pairs from .env (values are never printed)."""
    env = {}
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip().strip('"').strip("'")
            if v:
                env[k.strip()] = v
    for k in ("LINZ_API_KEY", "LRIS_API_KEY", "LINZ_BASEMAPS_KEY", "MFE_API_KEY"):
        if k not in env and os.environ.get(k):
            env[k] = os.environ[k]
    return env


def load_fuels() -> list[dict]:
    rows = []
    with open(ROOT / "fuels.csv", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            r["lcdb_class"] = int(r["lcdb_class"])
            r["ros_mult"] = float(r["ros_mult"])
            r["curing_pct"] = float(r["curing_pct"])
            r["load_t_ha"] = float(r["load_t_ha"])
            rows.append(r)
    return rows


def fuel_rows(cfg: dict | None = None) -> list[dict]:
    """fuels.csv plus one extra row for irrigated pasture (always last)."""
    cfg = cfg or load_config()
    rows = load_fuels()
    rows.append({"lcdb_class": -1, "lcdb_name": "Irrigated pasture", "fuel": "irrigated",
                 "model": "o1a", "ros_mult": 1.0,
                 "curing_pct": float(cfg["spread"]["irrigated_curing_pct"]),
                 "load_t_ha": 3.0})
    return rows


def grass_rows(fuels: list[dict]) -> set[int]:
    return {i for i, f in enumerate(fuels) if f["model"] in ("o1a", "o1b")}


# ----------------------------------------------------------------------------
# Fire Weather Index pieces
# ----------------------------------------------------------------------------
def ffmc_from_weather(temp_c: float, rh_pct: float) -> float:
    """FFMC of fine fuel at equilibrium (drying) moisture for the given air.

    Uses the FWI System's drying equilibrium moisture content (Van Wagner 1987).
    Assumes fuels have had time to dry to equilibrium, i.e. no recent rain.
    """
    h, t = rh_pct, temp_c
    ed = (0.942 * h ** 0.679 + 11.0 * math.exp((h - 100.0) / 10.0)
          + 0.18 * (21.1 - t) * (1.0 - math.exp(-0.115 * h)))
    ed = max(ed, 2.0)
    return 59.5 * (250.0 - ed) / (147.2 + ed)


def f_ffmc(ffmc: float) -> float:
    m = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
    return 91.9 * math.exp(-0.1386 * m) * (1.0 + m ** 5.31 / 4.93e7)


def f_wind(w):
    """Wind function of the ISI, with the FBP high-wind form above 40 km/h."""
    w = np.asarray(w, dtype=np.float64)
    return np.where(w < 40.0, np.exp(0.05039 * w),
                    12.0 * (1.0 - np.exp(-0.0818 * (w - 28.0))))


def isi(wind_kmh, ffmc: float):
    return 0.208 * f_wind(wind_kmh) * f_ffmc(ffmc)


# ----------------------------------------------------------------------------
# Rate-of-spread models (head fire, flat ground), m/min as a function of ISI
# ----------------------------------------------------------------------------
def _fbp(isi_v, a, b, c):
    return a * (1.0 - np.exp(-b * isi_v)) ** c


def curing_factor(curing_pct: float) -> float:
    """Grass curing factor as tabulated in the NZ manual (0 at 50%, 1 at 100%)."""
    return float(min(1.0, max(0.0, 0.02 * curing_pct - 1.0)))


def ros_model(model: str, isi_v, curing_pct: float = 100.0):
    isi_v = np.asarray(isi_v, dtype=np.float64)
    if model == "none":
        return np.zeros_like(isi_v)
    if model == "o1a":      # grazed / cut grass
        return _fbp(isi_v, 190.0, 0.0310, 1.4) * curing_factor(curing_pct)
    if model == "o1b":      # standing / rank grass, tussock
        return _fbp(isi_v, 250.0, 0.0350, 1.7) * curing_factor(curing_pct)
    if model == "scrub":    # curve fitted to the manual's scrubland table
        return _fbp(isi_v, 82.0, 0.1, 1.5)
    if model == "c6":       # conifer plantation, surface fire, BUI effect = 1
        return _fbp(isi_v, 30.0, 0.0800, 3.0)
    if model == "d1":       # leafless deciduous
        return _fbp(isi_v, 30.0, 0.0232, 1.6)
    if model == "m2":       # mixedwood, 60% conifer (C-2) / 40% deciduous
        return (0.6 * _fbp(isi_v, 110.0, 0.0282, 1.5)
                + 0.2 * 0.4 * _fbp(isi_v, 30.0, 0.0232, 1.6))
    raise ValueError(f"unknown spread model {model!r}")


ISI_GRID = np.arange(0.0, 150.0001, 0.05)


def ros_table(fuels: list[dict]) -> np.ndarray:
    """ROS lookup, one row per fuels.csv row, columns follow ISI_GRID."""
    tab = np.zeros((len(fuels), ISI_GRID.size))
    for i, f in enumerate(fuels):
        tab[i] = ros_model(f["model"], ISI_GRID, f["curing_pct"]) * f["ros_mult"]
    return tab


def slope_factor(slope_pct, max_slope_pct: float = 70.0):
    s = np.minimum(np.asarray(slope_pct, dtype=np.float64), max_slope_pct)
    return np.exp(3.533 * (s / 100.0) ** 1.2)


def length_to_breadth(wind_kmh, is_grass):
    w = np.asarray(wind_kmh, dtype=np.float64)
    grass = np.where(w >= 1.0, 1.1 * np.maximum(w, 1.0) ** 0.464, 1.0)
    other = 1.0 + 8.729 * (1.0 - np.exp(-0.030 * w)) ** 2.155
    return np.maximum(1.0, np.where(is_grass, grass, other))


def spread_field(fuel_row, tab, grass_rows, gx, gy, wind_kmh, wind_to_deg,
                 ffmc, factor=None, max_slope_pct=70.0):
    """Directional spread field for one block of cells.

    fuel_row : int array, row of `tab` for each cell
    gx, gy   : terrain gradient dz/d(east), dz/d(north)
    wind_kmh : per-cell 10 m wind speed
    wind_to_deg : compass bearing the wind blows towards
    Returns head ROS `h` and the ellipse terms (a, ex, ey) such that the ROS
    in unit direction (ux, uy) is  a / (1 - ex*ux - ey*uy).

    Wind and slope are combined the FBP way: slope is converted to an
    equivalent wind speed blowing upslope and added to the wind as a vector.
    """
    shape = fuel_row.shape
    ff = f_ffmc(ffmc)
    isz = 0.208 * ff
    g = np.hypot(gx, gy)
    sf = slope_factor(100.0 * g, max_slope_pct)
    gsafe = np.where(g > 1e-6, g, 1.0)
    upx, upy = gx / gsafe, gy / gsafe

    wse = np.zeros(shape, dtype=np.float64)
    rows = np.unique(fuel_row)
    eps = 1e-9 * np.arange(ISI_GRID.size)
    for r in rows:
        if tab[r, -1] <= 0:
            continue
        m = fuel_row == r
        rsz = np.interp(isz, ISI_GRID, tab[r])
        if rsz <= 0:
            continue
        isf = np.interp(rsz * sf[m], tab[r] + eps, ISI_GRID)
        wse[m] = np.maximum(0.0, np.log(np.maximum(isf, isz) / isz) / 0.05039)

    th = math.radians(wind_to_deg)
    vx = wind_kmh * math.sin(th) + wse * upx
    vy = wind_kmh * math.cos(th) + wse * upy
    wsv = np.hypot(vx, vy)
    wsafe = np.where(wsv > 1e-6, wsv, 1.0)
    ux, uy = vx / wsafe, vy / wsafe

    isi_v = isz * f_wind(wsv)
    h = np.zeros(shape, dtype=np.float64)
    is_grass = np.zeros(shape, dtype=bool)
    for r in rows:
        if tab[r, -1] <= 0:
            continue
        m = fuel_row == r
        h[m] = np.interp(isi_v[m], ISI_GRID, tab[r])
        if r in grass_rows:
            is_grass[m] = True
    if factor is not None:
        h *= factor
    lb = length_to_breadth(wsv, is_grass)
    e = np.sqrt(1.0 - 1.0 / (lb * lb))
    a = h * (1.0 - e)
    return (h.astype(np.float32), a.astype(np.float32),
            (e * ux).astype(np.float32), (e * uy).astype(np.float32))


# ----------------------------------------------------------------------------
# Minimum-travel-time fire growth (Dijkstra on a 16-neighbour grid)
# ----------------------------------------------------------------------------
# (drow, dcol) for 8 neighbours then 8 knight moves
_DR = np.array([-1, 1, 0, 0, -1, -1, 1, 1, -2, -2, 2, 2, -1, 1, -1, 1], dtype=np.int64)
_DC = np.array([0, 0, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -2, -2, 2, 2], dtype=np.int64)


@njit(cache=True)
def _push(hk, hv, n, key, val):
    if n >= hk.size:
        hk2 = np.empty(hk.size * 2, dtype=hk.dtype)
        hv2 = np.empty(hv.size * 2, dtype=hv.dtype)
        hk2[:n] = hk[:n]
        hv2[:n] = hv[:n]
        hk, hv = hk2, hv2
    i = n
    while i > 0:
        p = (i - 1) >> 1
        if hk[p] <= key:
            break
        hk[i] = hk[p]
        hv[i] = hv[p]
        i = p
    hk[i] = key
    hv[i] = val
    return hk, hv, n + 1


@njit(cache=True)
def _pop(hk, hv, n):
    key = hk[0]
    val = hv[0]
    n -= 1
    if n > 0:
        lk = hk[n]
        lv = hv[n]
        i = 0
        while True:
            c = 2 * i + 1
            if c >= n:
                break
            if c + 1 < n and hk[c + 1] < hk[c]:
                c += 1
            if hk[c] >= lk:
                break
            hk[i] = hk[c]
            hv[i] = hv[c]
            i = c
        hk[i] = lk
        hv[i] = lv
    return key, val, n


@njit(cache=True)
def burn(seeds, a, ex, ey, cell, tmax, t, dr, dc):
    """Grow a fire from `seeds` (flat cell indices) for `tmax` minutes.

    `t` must be all +inf on entry; on exit it holds arrival times for every
    cell touched. Returns the flat indices of cells burned within tmax, in
    order of arrival.
    """
    nrow, ncol = a.shape
    hk = np.empty(4096, dtype=np.float32)
    hv = np.empty(4096, dtype=np.int64)
    n = 0
    touched = np.empty(4096, dtype=np.int64)
    nt = 0
    burned = np.empty(4096, dtype=np.int64)
    nb = 0
    tf = t.ravel()
    for s in seeds:
        r0 = s // ncol
        c0 = s % ncol
        if a[r0, c0] > 0.0 and tf[s] > 0.0:
            tf[s] = 0.0
            hk, hv, n = _push(hk, hv, n, np.float32(0.0), s)
            if nt >= touched.size:
                tmp = np.empty(touched.size * 2, dtype=np.int64)
                tmp[:nt] = touched[:nt]
                touched = tmp
            touched[nt] = s
            nt += 1
    while n > 0:
        key, idx, n = _pop(hk, hv, n)
        if key > tf[idx]:
            continue
        if nb >= burned.size:
            tmp = np.empty(burned.size * 2, dtype=np.int64)
            tmp[:nb] = burned[:nb]
            burned = tmp
        burned[nb] = idx
        nb += 1
        r = idx // ncol
        c = idx % ncol
        ai = a[r, c]
        exi = ex[r, c]
        eyi = ey[r, c]
        for k in range(16):
            rr = r + dr[k]
            cc = c + dc[k]
            if rr < 0 or rr >= nrow or cc < 0 or cc >= ncol:
                continue
            aj = a[rr, cc]
            if aj <= 0.0:
                continue
            if k >= 8:
                # knight move: both cells it passes between must be burnable
                if abs(dr[k]) == 2:
                    r1 = r + dr[k] // 2
                    if a[r1, c] <= 0.0 or a[r1, cc] <= 0.0:
                        continue
                else:
                    c1 = c + dc[k] // 2
                    if a[r, c1] <= 0.0 or a[rr, c1] <= 0.0:
                        continue
            elif k >= 4:
                # diagonal: do not squeeze between two blocked cells
                if a[rr, c] <= 0.0 and a[r, cc] <= 0.0:
                    continue
            dx = dc[k] * cell
            dy = -dr[k] * cell
            d = math.sqrt(dx * dx + dy * dy)
            ux = dx / d
            uy = dy / d
            inv_i = (1.0 - exi * ux - eyi * uy) / ai
            inv_j = (1.0 - ex[rr, cc] * ux - ey[rr, cc] * uy) / aj
            # float32 throughout so a stored time compares equal to itself
            tn = np.float32(key + 0.5 * d * (inv_i + inv_j))
            if tn > tmax:
                continue
            j = rr * ncol + cc
            if tn < tf[j]:
                if tf[j] == np.inf:
                    if nt >= touched.size:
                        tmp = np.empty(touched.size * 2, dtype=np.int64)
                        tmp[:nt] = touched[:nt]
                        touched = tmp
                    touched[nt] = j
                    nt += 1
                tf[j] = tn
                hk, hv, n = _push(hk, hv, n, tn, j)
    return burned[:nb], touched[:nt]


@njit(parallel=True, cache=True)
def simulate(seed_idx, tresp, a, ex, ey, cell, tmax, pcl, bld, bldc, tstack,
             counts, dr, dc):
    """One fire per ignition cell. Returns per-ignition tallies.

    tresp  : minutes until the first fire crew could arrive at each ignition
    tstack : (threads, rows, cols) float32, all inf (scratch, restored to inf)
    counts : (threads, rows, cols) uint16, incremented for every burned cell
    Output columns: cells burned, conservation cells burned, buildings,
    commercial buildings, minutes to first building (inf if none), cells
    burned when the crew arrives, buildings reached when the crew arrives.
    """
    nthreads = tstack.shape[0]
    nign = seed_idx.size
    out = np.zeros((nign, 7), dtype=np.float32)
    pf = pcl.ravel()
    bf = bld.ravel()
    bcf = bldc.ravel()
    for th in prange(nthreads):
        t = tstack[th]
        tf = t.ravel()
        cf = counts[th].ravel()
        seeds = np.empty(1, dtype=np.int64)
        for i in range(th, nign, nthreads):
            if seed_idx[i] < 0:
                out[i, 4] = np.inf
                continue
            seeds[0] = seed_idx[i]
            burned, touched = burn(seeds, a, ex, ey, cell, tmax, t, dr, dc)
            npcl = 0
            nbld = 0
            nbldc = 0
            tfirst = np.inf
            narr = 0
            nbarr = 0
            for j in burned:
                cf[j] += 1
                if tf[j] <= tresp[i]:
                    narr += 1
                    nbarr += bf[j]
                if pf[j]:
                    npcl += 1
                if bf[j] > 0:
                    nbld += bf[j]
                    nbldc += bcf[j]
                    if tf[j] < tfirst:
                        tfirst = tf[j]
            out[i, 0] = burned.size
            out[i, 1] = npcl
            out[i, 2] = nbld
            out[i, 3] = nbldc
            out[i, 4] = tfirst
            out[i, 5] = narr
            out[i, 6] = nbarr
            for j in touched:
                tf[j] = np.inf
    return out


def neighbours():
    return _DR.copy(), _DC.copy()


# ----------------------------------------------------------------------------
# Raster IO
# ----------------------------------------------------------------------------
def grid_profile():
    """Analysis grid written by 02_grids (transform, shape, CRS)."""
    import json
    from affine import Affine
    g = json.loads((WORK / "grid.json").read_text())
    return g, Affine(*g["transform"])


def write_tif(name: str, arr: np.ndarray, nodata=None):
    import rasterio
    g, tr = grid_profile()
    WORK.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        WORK / name, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
        count=1, dtype=arr.dtype, crs="EPSG:2193", transform=tr, nodata=nodata,
        compress="deflate", tiled=True, blockxsize=512, blockysize=512,
        BIGTIFF="IF_SAFER",
    ) as dst:
        dst.write(arr, 1)


def read_tif(name: str) -> np.ndarray:
    import rasterio
    with rasterio.open(WORK / name) as src:
        return src.read(1)
