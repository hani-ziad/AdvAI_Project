"""One row per (sprint, checkpoint) with every feature set and both labels.

Checkpoints are 25% and 50% of the planned sprint length. A sprint already
closed at a checkpoint is dropped there (its outcome is known), as in Phase 3.

Feature sets (all use only information time-stamped at or before the checkpoint):
  paper     R1-R11 from the burndown so far (paper_features.py)
  phase1    the Assignment 1 planning-time features (LEGACY_NUMERIC) + project
  state     Phase 3 sprint-state features at the checkpoint (PROGRESS_FEATURES)
  history   Phase 2 strict history and capacity features (STRICT_NUMERIC)

Labels come from the whole sprint chart, so they are only used as targets:
  tag_main  Poor/Fair/Good/Excellent from 0.5*pace + 0.5*final completion
  tag_pace  the same from pace only (sensitivity)

    python -m advai.build_dataset
"""
from __future__ import annotations

import pandas as pd

import advai
from advai.paper_features import R_COLS, add_tag, build_table, pace_score, r_features  # noqa: F401
from phase2.data import LEGACY_NUMERIC, STRICT_NUMERIC, load_phase2_frame, population, time_ordered_split
from phase3.data import PROGRESS_FEATURES, add_progress_features, build_checkpoint_state

CHECKPOINTS = (0.25, 0.5)
OUT = advai.DATA / "early_warning_frame.csv"


def paper_features_at(daily: pd.DataFrame, planned_days: pd.Series, frac: float) -> pd.DataFrame:
    """R1-R11 using only the days up to frac of the PLANNED length (same clock as Phase 3)."""
    rows = []
    for sid, g in daily.groupby("sprint_id", sort=False):
        g = g.sort_values("day").copy()
        g["n_days"] = planned_days[sid]
        rows.append({"sprint_id": sid, "checkpoint": frac, **r_features(g, frac)})
    return pd.DataFrame(rows)


def build() -> pd.DataFrame:
    daily = pd.read_csv(advai.DATA / "burndown_daily.csv", parse_dates=["start_date"])

    # labels from the whole chart, cut points fitted on training sprints
    end = build_table(daily, 1.0)
    lab_main, _ = add_tag(end, "score")
    lab_pace, _ = add_tag(end, "pace")
    labels = pd.DataFrame({"sprint_id": end["sprint_id"], "tag_main": lab_main["TAG"],
                           "tag_pace": lab_pace["TAG"]})

    # Phase 3 state at the checkpoints, merged with the Phase 2 frame (history, legacy)
    base = load_phase2_frame()
    ne = population(base, "nonempty")
    state = build_checkpoint_state(CHECKPOINTS, cohort_ids=ne["sprint_id"])
    d = add_progress_features(state, ne.drop(columns=[c for c in ("end_date", "complete_date") if c in ne]))
    tr, te = time_ordered_split(ne)
    side = pd.concat([pd.Series("train", index=tr["sprint_id"]), pd.Series("test", index=te["sprint_id"])])
    d["split"] = d["sprint_id"].map(side)
    planned = ((d["end_date"] - d["start_date"]).dt.total_seconds() / 86400.0).groupby(d["sprint_id"]).first()

    paper = pd.concat([paper_features_at(daily, planned, f) for f in CHECKPOINTS])
    keep = (["sprint_id", "checkpoint", "project", "start_date", "split", "open_at_checkpoint"]
            + list(dict.fromkeys(LEGACY_NUMERIC + STRICT_NUMERIC + PROGRESS_FEATURES)))
    out = (d[keep].merge(paper, on=["sprint_id", "checkpoint"], how="inner")
           .merge(labels, on="sprint_id", how="inner"))
    out = out[out["open_at_checkpoint"]].drop(columns="open_at_checkpoint")
    return out.sort_values(["checkpoint", "project", "start_date", "sprint_id"]).reset_index(drop=True)


if __name__ == "__main__":
    f = build()
    f.to_csv(OUT, index=False)
    print(f.groupby("checkpoint").agg(n=("sprint_id", "size"), train=("split", lambda s: (s == "train").sum()),
                                      poor_main=("tag_main", lambda s: (s == 0).mean()),
                                      poor_pace=("tag_pace", lambda s: (s == 0).mean())).round(3))
    print(f"-> {OUT}")
