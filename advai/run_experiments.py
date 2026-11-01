"""Early warning of Poor sprints: our model vs the paper's method vs Assignment 1.

Follows report/plan.md. For each label (main, pace-only), checkpoint (25%,
50%) and split (time split, leave one project out) it scores every model on
"will this sprint end Poor?", then bootstraps the 8 projects 2,000 times.

    python -m advai.run_experiments
"""
from __future__ import annotations

import json
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import advai
from advai.paper_features import R_COLS
from phase2.data import LEGACY_NUMERIC, STRICT_NUMERIC
from phase2.models import feature_columns, make_classifier
from phase3.data import PROGRESS_FEATURES

SEED = advai.SEED
N_BOOT = 2000
OURS_NUMERIC = list(dict.fromkeys(R_COLS + PROGRESS_FEATURES + STRICT_NUMERIC))
LABELS = {"main": "tag_main", "pace only": "tag_pace"}


# ---- models: each returns (fit on train frame with 4-class tag y4) -> Poor score on test --------
def paper_svm():
    """The paper's method: SVM on R1-R11, trained on the 4 classes; Poor probability as the score."""
    est = make_pipeline(StandardScaler(), SVC(probability=True, random_state=SEED))

    def run(tr, te, y4):
        m = clone(est).fit(tr[R_COLS], y4)
        return m.predict_proba(te[R_COLS])[:, list(m.classes_).index(0)], m.predict(te[R_COLS])
    return run


def binary_and_4class(name, numeric, with_project=True):
    """A Phase 1/2 classifier trained for Poor vs rest (score) and, separately, for the 4 classes."""
    cols = feature_columns(numeric, with_project)

    def run(tr, te, y4):
        mb = make_classifier(name, numeric, with_project).fit(tr[cols], (y4 == 0).astype(int))
        m4 = make_classifier(name, numeric, with_project).fit(tr[cols], y4)
        return mb.predict_proba(te[cols])[:, 1], m4.predict(te[cols])
    return run


def pct_done(tr, te, y4):
    """No learning: the less is done so far, the higher the risk."""
    return -te["R1"].to_numpy(float), None


MODELS = {
    "% done so far (no learning)": pct_done,
    "Paper: SVM on R1-R11": paper_svm(),
    "Assignment 1 winner: RF, planning-time": binary_and_4class("random_forest", LEGACY_NUMERIC),
    "GB on R1-R11 only (same algorithm, paper features)": binary_and_4class("gradient_boosting", R_COLS, with_project=False),
    "Ours: GB, all features": binary_and_4class("gradient_boosting", OURS_NUMERIC),
    "Ours: RF, all features": binary_and_4class("random_forest", OURS_NUMERIC),
}
OURS = "Ours: GB, all features"
COMPARE = ["Paper: SVM on R1-R11", "GB on R1-R11 only (same algorithm, paper features)",
           "Assignment 1 winner: RF, planning-time", "% done so far (no learning)"]


# ---- metrics ----------------------------------------------------------------------------------
def prec_at_recall(y, s, target=0.8):
    p, r, _ = precision_recall_curve(y, s)
    return float(p[r >= target].max())


def metrics(y, s):
    return {"pr_auc": average_precision_score(y, s), "roc_auc": roc_auc_score(y, s),
            "prec_at_80_recall": prec_at_recall(y, s)}


def cluster_boot(proj, y, scores: dict, rng):
    """Resample projects with replacement; PR-AUC of each model and of ours minus each comparator."""
    groups = {p: np.flatnonzero(proj == p) for p in np.unique(proj)}
    keys = list(groups)
    out = {k: [] for k in scores}
    diffs = {c: [] for c in COMPARE}
    for _ in range(N_BOOT):
        idx = np.concatenate([groups[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        ap = {k: average_precision_score(yy, s[idx]) for k, s in scores.items()}
        for k in scores:
            out[k].append(ap[k])
        for c in COMPARE:
            diffs[c].append(ap[OURS] - ap[c])
    ci = lambda v: tuple(round(float(x), 3) for x in np.percentile(v, [2.5, 97.5]))  # noqa: E731
    return {k: ci(v) for k, v in out.items()}, {c: ci(v) for c, v in diffs.items()}


# ---- runs -------------------------------------------------------------------------------------
def predict(frame, label_col, split):
    """Pooled test predictions for every model: (test frame, {model: score}, {model: 4-class pred})."""
    if split == "time split":
        folds = [(frame[frame["split"] == "train"], frame[frame["split"] == "test"])]
    else:
        folds = [(frame[frame["project"] != p], frame[frame["project"] == p])
                 for p in sorted(frame["project"].unique())]
    tests, scores, preds = [], {m: [] for m in MODELS}, {m: [] for m in MODELS}
    for tr, te in folds:
        tests.append(te)
        y4 = tr[label_col].to_numpy()
        for m, run in MODELS.items():
            s, p = run(tr, te, y4)
            scores[m].append(s)
            preds[m].append(p)
    te = pd.concat(tests)
    return (te, {m: np.concatenate(v) for m, v in scores.items()},
            {m: (np.concatenate(v) if v[0] is not None else None) for m, v in preds.items()})


def main():
    frame = pd.read_csv(advai.DATA / "early_warning_frame.csv", parse_dates=["start_date"])
    rng = np.random.default_rng(SEED)
    rows, diff_rows = [], []
    for label, col in LABELS.items():
        for cp in sorted(frame["checkpoint"].unique()):
            f = frame[frame["checkpoint"] == cp]
            for split in ("time split", "leave one project out"):
                te, scores, preds = predict(f, col, split)
                y = (te[col].to_numpy() == 0).astype(int)
                y4 = te[col].to_numpy()
                ci, dci = cluster_boot(te["project"].to_numpy(), y, scores, rng)
                base = {"label": label, "checkpoint": cp, "split": split, "n_test": len(te),
                        "poor_rate": round(float(y.mean()), 3)}
                for m, s in scores.items():
                    r = {**base, "model": m, **{k: round(v, 3) for k, v in metrics(y, s).items()},
                         "pr_auc_lo": ci[m][0], "pr_auc_hi": ci[m][1]}
                    r["macro_f1_4class"] = (round(f1_score(y4, preds[m], average="macro"), 3)
                                            if preds[m] is not None else None)
                    rows.append(r)
                ours = average_precision_score(y, scores[OURS])
                for c in COMPARE:
                    diff_rows.append({**base, "comparison": f"{OURS} minus {c}",
                                      "pr_auc_diff": round(ours - average_precision_score(y, scores[c]), 3),
                                      "ci_lo": dci[c][0], "ci_hi": dci[c][1],
                                      "ci_excludes_zero": bool(dci[c][0] > 0 or dci[c][1] < 0)})
                print(label, cp, split, "done", flush=True)

    res, dif = pd.DataFrame(rows), pd.DataFrame(diff_rows)
    advai.RESULTS.mkdir(exist_ok=True)
    res.to_csv(advai.RESULTS / "early_warning_results.csv", index=False)
    dif.to_csv(advai.RESULTS / "early_warning_differences.csv", index=False)
    summary = {"rows": len(frame), "by_checkpoint": frame.groupby("checkpoint").size().to_dict(),
               "ours_features": len(OURS_NUMERIC) + 1, "bootstrap": N_BOOT, "seed": SEED,
               "python": platform.python_version(), "sklearn": sklearn.__version__,
               "pandas": pd.__version__, "numpy": np.__version__}
    (advai.RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print(res.to_string(index=False))
    print(dif.to_string(index=False))


if __name__ == "__main__":
    main()
