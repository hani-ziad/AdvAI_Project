"""Checks carried over from the CCECE audit of 8 Nov 2026.

The CCECE sprint-risk audit found two things that apply here:
  1. Pooled scores reward telling projects apart. Late closure scored 0.751 pooled but 0.634
     within a project, because history features mostly identify the team.
  2. A plain count can match a model. For spillover, the number of committed issues still
     open tied the in-sprint model.

This script reruns the main comparison (Confluence Cloud left out, refit grade, exactly as in
run_without_confluence_cloud.py) and adds:
  - within-project scores: ROC-AUC and PR-AUC worked out inside each project, then averaged
    over the projects that have both Poor and non-Poor sprints;
  - two more no-learning rules: "issues still open" (the count of committed issues not done)
    and "nothing finished yet" (no committed issue done so far);
  - for each comparison of my model with another, the within-project ROC-AUC difference with
    a 95% interval from resampling sprints inside each project (1,000 times).
It first checks that the pooled PR-AUCs match results/no_confluence_cloud_results.csv.

    python -m advai.audit_checks        (about 5 minutes)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import advai
from advai.run_experiments import LABELS, MODELS, OURS, SEED, predict
from advai.run_without_confluence_cloud import EXCLUDE, refit_labels

N_BOOT = 1000
RULES = {
    "Issues still open (no learning)": lambda te: te["remaining_n"].to_numpy(float),
    "Nothing finished yet (no learning)": lambda te: (te["done_n"].to_numpy(float) == 0).astype(float),
}
VERSUS = ["% done so far (no learning)", "Issues still open (no learning)",
          "Nothing finished yet (no learning)", "Paper: SVM on R1-R11",
          "GB on R1-R11 only (same algorithm, paper features)"]


def within(proj, y, s, metric):
    vals = [metric(y[proj == p], s[proj == p]) for p in np.unique(proj)
            if 0 < y[proj == p].sum() < (proj == p).sum()]
    return float(np.mean(vals)), len(vals)


def within_boot(proj, y, a, b, rng):
    groups = [np.flatnonzero(proj == p) for p in np.unique(proj)]
    out = []
    for _ in range(N_BOOT):
        d = []
        for g in groups:
            gi = g[rng.integers(0, len(g), len(g))]
            if 0 < y[gi].sum() < len(gi):
                d.append(roc_auc_score(y[gi], a[gi]) - roc_auc_score(y[gi], b[gi]))
        out.append(np.mean(d))
    return [round(float(x), 3) for x in np.percentile(out, [2.5, 97.5])]


def main():
    frame = pd.read_csv(advai.DATA / "early_warning_frame.csv", parse_dates=["start_date"])
    daily = pd.read_csv(advai.DATA / "burndown_daily.csv", parse_dates=["start_date"])
    kept = frame[frame["project"] != EXCLUDE].copy()
    labels, _ = refit_labels(daily)
    f_all = kept.drop(columns=["tag_main", "tag_pace"]).merge(labels, on="sprint_id", how="inner")
    saved = pd.read_csv(advai.RESULTS / "no_confluence_cloud_results.csv")
    saved = saved[saved["version"] == "refit grade"]

    rng = np.random.default_rng(SEED)
    rows, diffs, mismatches = [], [], []
    for label, col in LABELS.items():
        for cp in sorted(f_all["checkpoint"].unique()):
            f = f_all[f_all["checkpoint"] == cp]
            for split in ("time split", "leave one project out"):
                te, scores, _ = predict(f, col, split)
                for name, rule in RULES.items():
                    scores[name] = rule(te)
                y = (te[col].to_numpy() == 0).astype(int)
                proj = te["project"].to_numpy()
                for m, s in scores.items():
                    pooled_pr = average_precision_score(y, s)
                    if m in MODELS:
                        ref = saved[(saved["label"] == label) & (saved["checkpoint"] == cp)
                                    & (saved["split"] == split) & (saved["model"] == m)]["pr_auc"]
                        if len(ref) != 1 or abs(ref.item() - pooled_pr) > 0.0015:
                            mismatches.append((label, cp, split, m, round(pooled_pr, 3),
                                               None if len(ref) != 1 else ref.item()))
                    w_roc, n_proj = within(proj, y, s, roc_auc_score)
                    w_pr, _ = within(proj, y, s, average_precision_score)
                    rows.append({"label": label, "checkpoint": cp, "split": split, "model": m,
                                 "n_test": len(te), "poor_rate": round(float(y.mean()), 3),
                                 "pooled_pr_auc": round(pooled_pr, 3),
                                 "pooled_roc_auc": round(roc_auc_score(y, s), 3),
                                 "within_roc_auc": round(w_roc, 3), "within_pr_auc": round(w_pr, 3),
                                 "projects_scored": n_proj})
                for c in VERSUS:
                    d = within(proj, y, scores[OURS], roc_auc_score)[0] - within(proj, y, scores[c], roc_auc_score)[0]
                    lo, hi = within_boot(proj, y, scores[OURS], scores[c], rng)
                    diffs.append({"label": label, "checkpoint": cp, "split": split,
                                  "comparison": f"{OURS} minus {c}", "within_roc_auc_diff": round(d, 3),
                                  "ci_lo": lo, "ci_hi": hi, "ci_excludes_zero": bool(lo > 0 or hi < 0)})
                print(label, cp, split, "done", flush=True)

    res, dif = pd.DataFrame(rows), pd.DataFrame(diffs)
    res.to_csv(advai.RESULTS / "audit_within_project_results.csv", index=False)
    dif.to_csv(advai.RESULTS / "audit_within_project_differences.csv", index=False)
    info = {"excluded_project": EXCLUDE, "grade": "refit", "bootstrap_within": N_BOOT, "seed": SEED,
            "pooled_pr_auc_matches_saved_results": not mismatches, "mismatches": mismatches}
    (advai.RESULTS / "audit_summary.json").write_text(json.dumps(info, indent=2, default=str))
    pd.set_option("display.width", 250)
    print(json.dumps(info, indent=2, default=str))
    print(res[res["label"] == "main"].to_string(index=False))
    print(dif[dif["label"] == "main"].to_string(index=False))


if __name__ == "__main__":
    main()
