import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import firelib as fl  # noqa: E402

CELL = 25.0
DR, DC = fl.neighbours()


def uniform(n, h, e, to_deg):
    th = math.radians(to_deg)
    a = np.full((n, n), h * (1 - e), dtype=np.float32)
    ex = np.full((n, n), e * math.sin(th), dtype=np.float32)
    ey = np.full((n, n), e * math.cos(th), dtype=np.float32)
    return a, ex, ey


def run(a, ex, ey, seed_rc, tmax):
    t = np.full(a.shape, np.inf, dtype=np.float32)
    seeds = np.array([seed_rc[0] * a.shape[1] + seed_rc[1]], dtype=np.int64)
    burned, _ = fl.burn(seeds, a, ex, ey, CELL, tmax, t, DR, DC)
    return t, burned


# ---- equations ------------------------------------------------------------
def test_grass_models_match_fbp_values():
    # O-1a / O-1b at full curing, hand-calculated from the published form
    assert fl.ros_model("o1a", 20.0, 100) == pytest.approx(190 * (1 - math.exp(-0.62)) ** 1.4)
    assert fl.ros_model("o1b", 20.0, 100) == pytest.approx(250 * (1 - math.exp(-0.70)) ** 1.7)
    assert fl.curing_factor(50) == 0 and fl.curing_factor(100) == 1
    assert fl.curing_factor(80) == pytest.approx(0.6)


def test_hardwood_is_half_scrub():
    fuels = fl.load_fuels()
    tab = fl.ros_table(fuels)
    by = {f["lcdb_class"]: i for i, f in enumerate(fuels)}
    assert np.allclose(tab[by[54]], 0.5 * tab[by[52]])


def test_slope_factor_matches_manual_table():
    # manual page F-2: upslope 5..35 degrees
    table = {5: 1.2, 10: 1.6, 15: 2.1, 20: 2.9, 25: 4.1, 30: 6.2, 35: 10.0}
    for deg, want in table.items():
        got = float(fl.slope_factor(100 * math.tan(math.radians(deg))))
        assert got == pytest.approx(want, rel=0.08), deg
    # capped beyond 70 %
    assert float(fl.slope_factor(150)) == pytest.approx(float(fl.slope_factor(70)))


def test_isi_reference_value():
    # FWI System: FFMC 90, wind 20 km/h -> ISI about 11.7
    assert float(fl.isi(20.0, 90.0)) == pytest.approx(11.7, abs=0.4)


def test_ffmc_hot_dry_is_high():
    assert 91 < fl.ffmc_from_weather(28, 20) < 97
    assert fl.ffmc_from_weather(15, 70) < 86


# ---- spread field ---------------------------------------------------------
def field(gx=0.0, gy=0.0, wind=30.0, to=135.0, cls=41):
    fuels = fl.load_fuels()
    tab = fl.ros_table(fuels)
    row = [i for i, f in enumerate(fuels) if f["lcdb_class"] == cls][0]
    grass = {i for i, f in enumerate(fuels) if f["model"] in ("o1a", "o1b")}
    shape = (3, 3)
    return fl.spread_field(
        np.full(shape, row), tab, grass, np.full(shape, gx), np.full(shape, gy),
        np.full(shape, wind), to, 92.0)


def test_flat_ground_spreads_downwind():
    h, a, ex, ey = field()
    e = math.hypot(ex[0, 0], ey[0, 0])
    # wind blowing towards the south-east
    assert ex[0, 0] > 0 and ey[0, 0] < 0
    assert h[0, 0] > 0 and 0.5 < e < 1
    assert a[0, 0] == pytest.approx(h[0, 0] * (1 - e), rel=1e-4)


def test_upslope_with_wind_is_faster_than_flat_and_downslope_slower():
    flat = field()[0][0, 0]
    # wind towards SE; terrain rising towards SE (gradient points SE)
    g = 0.3 / math.sqrt(2)
    up = field(gx=g, gy=-g)[0][0, 0]
    down = field(gx=-g, gy=g)[0][0, 0]
    assert up > flat > down


def test_slope_only_points_upslope():
    h, a, ex, ey = field(gx=0.4, gy=0.0, wind=0.0)
    assert ex[0, 0] > 0 and abs(ey[0, 0]) < 1e-6
    assert h[0, 0] > field(wind=0.0)[0][0, 0]


def test_non_fuel_does_not_burn():
    assert field(cls=20)[0][0, 0] == 0


# ---- fire growth ----------------------------------------------------------
def test_uniform_fire_is_the_expected_ellipse():
    h, e, n, tmax = 20.0, 0.8, 401, 120.0
    a, ex, ey = uniform(n, h, e, 90.0)          # wind towards the east
    c = n // 2
    t, burned = run(a, ex, ey, (c, c), tmax)
    reached = t <= tmax
    head = (np.where(reached[c])[0].max() - c) * CELL
    back = (c - np.where(reached[c])[0].min()) * CELL
    flank = (c - np.where(reached[:, c])[0].min()) * CELL
    assert head == pytest.approx(h * tmax, rel=0.05)
    assert back == pytest.approx(h * (1 - e) / (1 + e) * tmax, rel=0.12)
    assert flank == pytest.approx(h * (1 - e) * tmax, rel=0.12)   # r(90 deg)
    # area of the ellipse: pi * a * b, a = (head+back)/2, b = a*sqrt(1-e^2)
    semi = (h * tmax + h * (1 - e) / (1 + e) * tmax) / 2
    area = math.pi * semi * semi * math.sqrt(1 - e * e)
    # 16 directions cannot follow every bearing exactly, so the grid fire is
    # slightly small (about 10 % by area) and never larger than the ellipse
    assert 0.85 * area < burned.size * CELL * CELL <= 1.01 * area


def test_direction_is_respected_off_axis():
    h, e, n, tmax = 20.0, 0.8, 401, 100.0
    a, ex, ey = uniform(n, h, e, 135.0)         # towards the south-east
    c = n // 2
    t, _ = run(a, ex, ey, (c, c), tmax)
    rr, cc = np.where(t <= tmax)
    far = np.argmax(np.hypot(rr - c, cc - c))
    assert rr[far] > c and cc[far] > c          # south (rows down) and east
    assert math.hypot(rr[far] - c, cc[far] - c) * CELL == pytest.approx(h * tmax, rel=0.06)


def test_lake_blocks_and_fire_goes_round():
    n = 201
    a, ex, ey = uniform(n, 20.0, 0.7, 90.0)
    a[:, 120:124] = 0                           # full-height barrier
    c = n // 2
    t, _ = run(a, ex, ey, (c, c), 600.0)
    assert np.isinf(t[:, 124:]).all()
    a, ex, ey = uniform(n, 20.0, 0.7, 90.0)
    a[20:180, 120:124] = 0                      # gap at each end
    t, _ = run(a, ex, ey, (c, c), 2000.0)
    assert np.isfinite(t[c, 130])


def test_diagonal_barrier_one_cell_wide_holds():
    n = 101
    a, ex, ey = uniform(n, 20.0, 0.5, 90.0)
    for i in range(n):
        a[i, i] = 0                             # 8-connected diagonal line
    t, _ = run(a, ex, ey, (20, 60), 2000.0)     # start above the diagonal
    lower = np.tril(np.ones((n, n), bool), -1)
    assert np.isinf(t[lower]).all()


def test_stream_slows_but_does_not_stop():
    n = 201
    c = n // 2
    a0, ex, ey = uniform(n, 20.0, 0.7, 90.0)
    t0, _ = run(a0, ex, ey, (c, c), 600.0)
    a1 = a0.copy()
    a1[:, 120] *= 0.3
    t1, _ = run(a1, ex, ey, (c, c), 600.0)
    assert np.isfinite(t1[c, 150]) and t1[c, 150] > t0[c, 150]


def test_simulate_tallies_and_restores_scratch():
    n = 201
    a, ex, ey = uniform(n, 10.0, 0.6, 90.0)
    c = n // 2
    pcl = np.zeros((n, n), np.uint8)
    pcl[:, c + 10:] = 1
    bld = np.zeros((n, n), np.uint16)
    bldc = np.zeros((n, n), np.uint16)
    bld[c, c + 20] = 3
    bldc[c, c + 20] = 1
    tstack = np.full((2, n, n), np.inf, np.float32)
    counts = np.zeros((2, n, n), np.uint16)
    seeds = np.array([c * n + c, c * n + c, -1], dtype=np.int64)
    tresp = np.array([30.0, 60.0, 30.0], dtype=np.float32)
    out = fl.simulate(seeds, tresp, a, ex, ey, CELL, 120.0, pcl, bld, bldc,
                      tstack, counts, DR, DC)
    # fire is smaller when the crew arrives sooner; building at 50 min is
    # reached before a 60-minute arrival but not a 30-minute one
    assert 0 < out[0, 5] < out[1, 5] < out[1, 0]
    assert out[0, 6] == 0 and out[1, 6] == 3
    out[1, 5:] = out[0, 5:]
    assert np.isinf(tstack).all()
    assert out[0, 0] > 0 and np.array_equal(out[0], out[1])
    assert out[0, 2] == 3 and out[0, 3] == 1
    assert out[0, 4] == pytest.approx(20 * CELL / 10.0, rel=0.05)
    assert 0 < out[0, 1] < out[0, 0]
    assert out[2, 0] == 0 and np.isinf(out[2, 4])
    total = counts.sum(axis=0)
    assert total.max() == 2 and total.sum() == 2 * out[0, 0]
