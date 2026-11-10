"""Combine the simulation tallies into one patrol-priority score per line section.

For each ignition point, results are averaged across the wind scenarios
(weighted by how often each occurs), then four sub-scores (0-100) are formed:

  property      buildings reached (business-area buildings count extra)
  conservation  hectares of conservation land burned
  response      hectares already burning when the first fire crew arrives
  spread        total hectares burned

Each sub-score is log-scaled against the 99th-percentile ignition, so a
section must be about ten times worse to score twice as high, and a handful of
extreme sections do not flatten everything else. The score is the weighted sum
(weights in config.yaml). A line section takes the score of its worst point.

Outputs: data/work/priority.csv, outputs/patrol_priority.csv
"""
from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pandas as pd

import firelib as fl

CFG = fl.load_config()


def log_score(x: pd.Series) -> pd.Series:
    pos = x[x > 0]
    ref = float(np.percentile(pos, 99)) if len(pos) else 1.0
    return (100.0 * np.log1p(x) / np.log1p(ref)).clip(upper=100.0)


def main():
    p = CFG["priority"]
    w = p["weights"]
    assert abs(sum(w.values()) - 1.0) < 1e-6, "priority weights must sum to 1"
    scen = json.loads((fl.WORK / "scenarios.json").read_text())
    share = np.array([s["share"] for s in scen])
    share = share / share.sum()
    ign = pd.read_csv(fl.WORK / "ignitions.csv")

    def mean(col):
        return sum(ign[f"{col}_{s['id']}"] * sh for s, sh in zip(scen, share))

    k = float(p["commercial_multiplier"])
    ign["bld"] = mean("bld")
    ign["bldc"] = mean("bldc")
    ign["pcl_ha"] = mean("pcl_ha")
    ign["ha"] = mean("ha")
    ign["ha_arr"] = mean("ha_arr")
    ign["bld_arr"] = mean("bld_arr")
    ign["s_property"] = log_score(ign.bld + (k - 1.0) * ign.bldc)
    ign["s_conservation"] = log_score(ign.pcl_ha)
    ign["s_response"] = log_score(ign.ha_arr)
    ign["s_spread"] = log_score(ign.ha)
    ign["score"] = sum(w[c] * ign[f"s_{c}"] for c in w)

    # each line section takes its worst ignition point
    worst = ign.loc[ign.groupby("line_id")["score"].idxmax()].set_index("line_id")
    lines = gpd.read_file(fl.RAW / "lines.gpkg").set_index("line_id")
    cen = lines.geometry.centroid.to_crs(4326)
    out = pd.DataFrame({
        "voltage": lines.voltage, "length_m": lines.length_m,
        "lat": cen.y.round(5), "lon": cen.x.round(5)}).join(
        worst[["score", "s_property", "s_conservation", "s_response", "s_spread",
               "bld", "bldc", "pcl_ha", "ha", "resp_min", "ha_arr", "bld_arr"]])
    out = out.fillna(0).sort_values("score", ascending=False)
    out["rank"] = np.arange(1, len(out) + 1)
    cum = out.length_m.cumsum() / out.length_m.sum()
    edges, names, acc = [], [], 0.0
    for name, frac in p["bands"].items():
        acc += frac
        edges.append(acc)
        names.append(name)
    names.append("Low")
    out["band"] = [names[int(np.searchsorted(edges, c, side="left"))] if c <= edges[-1] else "Low"
                   for c in cum]
    for c in ("score", "s_property", "s_conservation", "s_response", "s_spread", "bld", "bldc", "bld_arr"):
        out[c] = out[c].round(1)
    for c in ("pcl_ha", "ha", "ha_arr", "resp_min"):
        out[c] = out[c].round(0)
    out.to_csv(fl.WORK / "priority.csv")

    pretty = out.reset_index().rename(columns={
        "line_id": "Line section", "rank": "Rank", "band": "Priority band",
        "score": "Priority score (0-100)", "voltage": "Voltage (V)", "length_m": "Length (m)",
        "lat": "Latitude", "lon": "Longitude",
        "s_property": "Property score", "s_conservation": "Conservation score",
        "s_response": "Response score", "s_spread": "Spread score",
        "bld": "Buildings reached", "bldc": "of which in business areas",
        "pcl_ha": "Conservation land burned (ha)", "ha": "Area burned (ha)",
        "resp_min": "First crew arrival (min)", "ha_arr": "Fire size at crew arrival (ha)",
        "bld_arr": "Buildings reached before crew arrival"})
    first = ["Rank", "Priority band", "Priority score (0-100)", "Line section"]
    pretty = pretty[first + [c for c in pretty.columns if c not in first]]
    outdir = fl.ROOT / "outputs"
    outdir.mkdir(exist_ok=True)
    pretty.to_csv(outdir / "patrol_priority.csv", index=False)

    print("scenario weights:", {s["id"]: round(float(sh), 2) for s, sh in zip(scen, share)})
    summ = out.groupby("band", sort=False).agg(sections=("score", "size"), km=("length_m", lambda v: v.sum() / 1000),
                                               min_score=("score", "min"), max_score=("score", "max"))
    print(summ.round(1).to_string())
    print("correlation of sub-scores with the total:",
          {c: round(float(out[f"s_{c}"].corr(out.score)), 2) for c in w})
    print(pretty.head(8)[["Rank", "Priority score (0-100)", "Latitude", "Longitude", "Buildings reached",
                          "Conservation land burned (ha)", "First crew arrival (min)",
                          "Fire size at crew arrival (ha)"]].to_string(index=False))


if __name__ == "__main__":
    main()
