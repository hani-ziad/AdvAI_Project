"""Copied from my paper critique (github.com/hani-ziad/AdvAI_critique).

My TAWOS versions of the paper's R1-R11 features and of its 4-level TAG.

The paper (Table 1) defines R1-R11 in words only and does not give the
thresholds behind "significant" or "excessive". The rules below are my
reading of those words, written down before I looked at any model result.
They work on issue counts, because TAWOS has no hours.

Per day k of a sprint there is scope (issues in the sprint), done (of those,
resolved) and remaining = scope - done. With n = planned length in days and
ideal = issues committed on day 0 / n:

  R1  % of work completed at the cut-off (end of sprint for the paper protocol)
  R2  any stall period?            a stall day = no new completed issue;
  R3  number of stall periods      a stall period = 2 or more stall days in a row
  R4  % of days inside stall periods
  R5  days inside stall periods
  R6  any day without significant reduction?  remaining fell by less than half the ideal daily rate
  R7  number of such days
  R8  any excessive reduction?     remaining fell by at least max(2, 25% of day-0 scope) in one day
  R9  number of excessive reductions
  R10 any significant increase?    remaining rose by at least max(1, 10% of day-0 scope) in one day
  R11 number of significant increases

TAG (my stand-in for the expert label). The paper's expert judged pace and
progress from the same two charts. I copy that idea with a fixed rule that
also reads the whole chart:

  pace  = mean over days of min(share done / share of time elapsed, 1)
  score = 0.5 * pace + 0.5 * share done at the end
  TAG   = Poor / Fair / Good / Excellent by cutting the score at the paper's
          class proportions (26.5%, 18.1%, 30.3%, 25.0%), with the cut points
          fitted on the training sprints only.

A second label, "pace only", cuts the pace term alone. I use it as a
sensitivity check, because my main label contains the final completion share
and so leans on R1 by construction.

So in my data the label is a known function of the chart. That is the point:
it shows how much of a high score is just the model re-reading the chart.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

R_COLS = [f"R{i}" for i in range(1, 12)]
CLASSES = ["Poor", "Fair", "Good", "Excellent"]
PAPER_SHARES = np.array([146, 100, 167, 138]) / 551  # class sizes from the paper's Table 4


def _runs(mask: np.ndarray, min_len: int = 2) -> tuple[int, int]:
    """Number of runs of True with length >= min_len, and the days they cover."""
    n_runs = days = run = 0
    for v in list(mask) + [False]:
        if v:
            run += 1
        else:
            if run >= min_len:
                n_runs += 1
                days += run
            run = 0
    return n_runs, days


def r_features(g: pd.DataFrame, upto_frac: float = 1.0) -> dict:
    """R1-R11 for one sprint, using only the days up to upto_frac of its length."""
    n_days = max(float(g["n_days"].iloc[0]), 1.0)
    g = g[g["elapsed_days"] <= upto_frac * n_days + 1e-9]
    scope = g["scope_n"].to_numpy(float)
    done = g["done_n"].to_numpy(float)
    rem = g["remaining_n"].to_numpy(float)
    scope0 = max(scope[0], 1.0)
    ideal = scope0 / n_days

    new_done = np.diff(done)
    drop = -np.diff(rem)  # positive = remaining went down
    steps = max(len(new_done), 1)

    n_stall, stall_days = _runs(new_done <= 0)
    no_red = (drop < 0.5 * ideal) & (rem[:-1] > 0)
    excessive = drop >= max(2.0, 0.25 * scope0)
    increase = -drop >= max(1.0, 0.10 * scope0)

    share_end = done[-1] / scope[-1] if scope[-1] > 0 else 0.0
    return {
        "R1": 100.0 * share_end,
        "R2": int(n_stall > 0), "R3": n_stall,
        "R4": 100.0 * stall_days / steps, "R5": stall_days,
        "R6": int(no_red.sum() > 0), "R7": int(no_red.sum()),
        "R8": int(excessive.sum() > 0), "R9": int(excessive.sum()),
        "R10": int(increase.sum() > 0), "R11": int(increase.sum()),
    }


def pace_score(g: pd.DataFrame) -> tuple[float, float]:
    """(pace, score) for one whole sprint (see module docstring)."""
    n_days = max(float(g["n_days"].iloc[0]), 1.0)
    g = g.iloc[1:]
    if g.empty:
        return 0.0, 0.0
    share = np.where(g["scope_n"] > 0, g["done_n"] / g["scope_n"].clip(lower=1), 0.0)
    elapsed = (g["elapsed_days"] / n_days).clip(lower=1e-6).to_numpy()
    pace = float(np.minimum(share / elapsed, 1.0).mean())
    return pace, 0.5 * pace + 0.5 * float(share[-1])


def build_table(daily: pd.DataFrame, upto_frac: float = 1.0) -> pd.DataFrame:
    rows = []
    for sid, g in daily.groupby("sprint_id", sort=False):
        g = g.sort_values("day")
        row = {"sprint_id": sid, "project": g["project"].iloc[0], "start_date": g["start_date"].iloc[0],
               "split": g["split"].iloc[0], "n_days": g["n_days"].iloc[0]}
        row.update(r_features(g, upto_frac))
        row["pace"], row["score"] = pace_score(g)
        rows.append(row)
    return pd.DataFrame(rows)


def add_tag(table: pd.DataFrame, column: str = "score") -> tuple[pd.DataFrame, np.ndarray]:
    """Cut a score into 4 classes at the paper's proportions, fitted on training sprints.

    column="score" is my main TAG. column="pace" is a sensitivity check that
    leaves the end-of-sprint completion out of the label, so R1 is less
    directly tied to it.
    """
    train_scores = table.loc[table["split"] == "train", column]
    cuts = np.quantile(train_scores, np.cumsum(PAPER_SHARES)[:-1])
    t = table.copy()
    t["TAG"] = np.digitize(t[column], cuts, right=True)  # 0 Poor .. 3 Excellent
    return t, cuts
