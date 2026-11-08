# AdvAI_Project: spotting Poor sprints early

My final project for CSBP711 Advanced AI, Fall 2026 (Option 1: close the gap from the paper critique).
Hani AbuSharkh, HCT Al Ain.

## The question

A quarter or halfway into a sprint, can we tell which sprints are going to end Poor? And does using what Jira already knows (the team's recent history, plus the sprint's current state and scope changes) do better than the burndown-only features of Pérez Castillo et al. (*Information*, 2024)?

Here is the gap I'm closing. The paper grades a sprint from 11 burndown features (R1 to R11) taken at the end of the sprint, but sells the method as a way to catch sprints that need "immediate intervention". In my critique (https://github.com/hani-ziad/AdvAI_critique), those same features on TAWOS dropped from .77 accuracy at the end of the sprint to .51 at the 25% mark. So this project builds a real early warning and tests it.

I wrote the plan before running anything (`report/plan.md`). After the first run I added one control model, gradient boosting on R1 to R11 only, and the plan explains why.

## Setup

- **Data:** TAWOS, the 1,691 sprints with committed work (the same group as Assignment 1). At each checkpoint I only keep sprints that are still open.
- **One project left out.** In Atlassian Confluence Cloud, 59% of the closed sprint issues have no resolution date (every other project is at or near 0%). The burndown counts an issue as done from its resolution date, so those sprints look like nothing was ever finished, and 76% of them get graded Poor. The main results therefore use the other 7 projects: 1,471 sprints at 25% and 1,460 at 50%. The original 8-project run is kept below for comparison.
- **Target:** whether the sprint ends Poor, using the grade from the critique, with the cut points refitted on the 7-project training sprints. A "pace only" grade is used as a check.
- **Splits:** the Assignment 1 time split (the first 70% of each project's sprints for training, the last 30% for testing), and leaving one project out. Seed 42.
- **Uncertainty:** a bootstrap over the projects (7 in the main results), 2,000 resamples.

| Model | Features | Why it's here |
|---|---|---|
| % done so far | R1 only, no learning | the simplest possible check |
| Paper: SVM on R1-R11 | the paper's 11 burndown features, trained on the 4 grades | the critique paper's method |
| GB on R1-R11 only | the paper's features, my algorithm | separates the algorithm from the features |
| Assignment 1 winner | the Phase 1 random forest on planning-time features + project | my Assignment 1 winner |
| **Mine: GB, all features** | R1-R11 + Phase 3 sprint state + Phase 2 history (37 numeric features) + project | the proposed model |
| Mine: RF, all features | same | a second check |

## Results

PR-AUC for "ends Poor" with the main grade, 7 projects. Random guessing gets .40 on the time-split test set (later sprints are Poor more often than earlier ones) and .30 across projects.

| Model | 25%, time split | 25%, across projects | 50%, time split | 50%, across projects |
|---|---|---|---|---|
| % done so far | .599 | .540 | .704 | .689 |
| Paper: SVM on R1-R11 | .599 | .466 | .711 | .630 |
| GB on R1-R11 only | .650 | .483 | .744 | .644 |
| Assignment 1 winner (RF) | .575 | .452 | .570 | .449 |
| **Mine: GB, all features** | **.680** | **.585** | **.757** | **.698** |
| Mine: RF, all features | .682 | .623 | .765 | .714 |

Mine minus the paper's method (PR-AUC difference, 95% project-bootstrap interval):

| | 25% | 50% |
|---|---|---|
| Time split | +.081 [.015, .135] | +.046 [.003, .108] |
| Across projects | +.119 [.056, .170] | +.068 [.010, .156] |

What this tells me:

- **Mine beats the paper's method at both checkpoints, but by much less than I first reported.** All four intervals sit above zero, but three of them start close to it (.003 to .015).
- **On the time split, most of that gain is the algorithm, not the Jira context.** Against gradient boosting on the paper's own features, mine is +.030 [−.013, .083] at 25% and +.012 [−.027, .058] at 50%. Neither is clear of zero.
- **Across projects, the Jira context does help.** Against the same control, mine is +.102 [.059, .150] at 25% and +.054 [.009, .088] at 50%. The extra features carry over to a team the model has never seen better than the burndown alone.
- **"% done" is a hard baseline.** At 50% across projects, mine is only +.009 [−.027, .096] above it. A Scrum Master reading the done count halfway through gets most of what my model gets.
- **Planning-time information isn't enough on its own.** The Assignment 1 winner is the weakest model and doesn't improve from 25% to 50%. Mine beats it in all 8 comparisons.
- **The alarms are a little cleaner.** If the model has to catch 80% of Poor sprints, 63% of my alarms are right at 25% and 72% at 50% (time split), against 58% and 68% for the paper's method.
- **The pace-only grade tells a stronger story.** 15 of its 16 comparisons are clear of zero. The one miss is mine against GB on R1-R11 at 50% on the time split.

### Scored within each project (added 8 Nov 2026)

My Software Engineering audit found that pooled scores reward telling projects apart: late closure scored .751 pooled but only .634 inside a single project. So I rescored the main comparison inside each project, with ROC-AUC averaged over the 7 projects, and added two more no-learning rules: "issues still open" (how many committed issues are not done) and "nothing finished yet". The script first checks that it reproduces every pooled PR-AUC above, and it does.

Within-project ROC-AUC, main grade, 7 projects:

| Model | 25%, time split | 25%, across projects | 50%, time split | 50%, across projects |
|---|---|---|---|---|
| % done so far | .742 | .792 | .847 | .886 |
| Nothing finished yet | .653 | .707 | .703 | .749 |
| Issues still open | .473 | .449 | .496 | .478 |
| Paper: SVM on R1-R11 | .734 | .783 | .824 | .872 |
| GB on R1-R11 only | .731 | .776 | .838 | .887 |
| **Mine: GB, all features** | **.745** | **.821** | **.850** | **.898** |

Mine minus each comparison, within projects (95% interval from resampling sprints inside each project):

| | 25%, time split | 25%, across projects | 50%, time split | 50%, across projects |
|---|---|---|---|---|
| vs % done | +.003 [−.032, .039] | +.030 [.013, .047] | +.004 [−.025, .033] | +.013 [.001, .023] |
| vs paper's SVM | +.012 [−.039, .062] | +.038 [.016, .061] | +.027 [−.010, .065] | +.026 [.011, .043] |
| vs GB on R1-R11 | +.014 [−.036, .063] | +.046 [.023, .068] | +.013 [−.020, .047] | +.011 [−.003, .024] |

What this changes:

- **Inside a project, on later sprints, my model is no better than "% done".** The gaps are +.003 and +.004 and both intervals cross zero. It is not clearly better than the paper's method either. Most of the pooled gain on the time split came from ranking whole projects, not sprints within them.
- **On a team the model has never seen, it keeps a small, real edge**: +.013 to +.030 over "% done" and +.026 to +.038 over the paper's method. That is the honest claim for this project.
- **Counting open issues is the wrong rule here.** It scores below chance, because a big sprint can have many issues open and still be on pace. Poor is about the share done, not the count, which is the opposite of what we saw for spillover.
- **"Nothing finished yet" is much weaker than "% done"** once each project is scored on its own.
- The pace-only grade gives my model clearer gains (18 of 20 intervals above zero, against 13 of 20 for the main grade), but that grade partly overlaps the early window, as my critique found, so I weigh it less.

Run it with `python -m advai.audit_checks` (about 10 minutes). Results: `results/audit_within_project_results.csv`, `results/audit_within_project_differences.csv` and `results/audit_summary.json`.

Full tables: `results/no_confluence_cloud_results.csv` and `results/no_confluence_cloud_differences.csv` (both versions, see below), and `results/no_confluence_cloud_summary.json`.

### What changed when Confluence Cloud was removed

My first run used all 8 projects. That run is still in `results/early_warning_results.csv` and `results/early_warning_differences.csv`. Its headline was mine minus the paper at +.189 [.064, .270] at 25% on the time split, and +.248 across projects. Without Confluence Cloud those gains roughly halve (+.081 and +.119).

The reason is that Confluence Cloud sprints were almost all Poor, and my model could see it coming from things the paper's SVM does not have: the project itself, and the team's past completion, which is near zero for that project because of the same missing dates. On the 25% time split, my model put the Confluence Cloud test sprints at the 82nd percentile of risk on average (the paper's SVM at the 67th), and they made up 30% of its top alarms. That was a data error, not early warning.

I ran two versions of the recheck. "Refit grade" removes the project before the grade is cut, so the cut points come from the 7 remaining projects. This is the one I report above. "Same grade" keeps the original labels and only drops the rows. It gives the same picture: mine is still ahead of the paper's method in 3 of 4 main-grade comparisons, and the gain over GB on R1-R11 is not clear of zero on the time split. Refitting moves the Poor flag for 9% of the remaining sprints.

## How to run

```
pip install -r requirements.txt
python -m advai.burndown          # about 15 s, writes data/burndown_daily.csv
python -m advai.build_dataset     # about 20 s, writes data/early_warning_frame.csv
python -m advai.run_experiments   # about 4 min, the original 8-project run, writes results/
python -m advai.run_without_confluence_cloud   # about 7 min, the 7-project recheck (main results)
python -m advai.audit_checks    # about 10 min, within-project scores and two more rules
```

Run these from the repository folder. My run used Python 3.13, scikit-learn 1.9.1, pandas 3.0 and numpy 2.5 (`results/summary.json`).

## Files

- `advai/` is the project code: the burndown rebuild and the R1-R11 features and grades (both copied from the critique), the dataset builder and the experiments.
- `pipeline/` holds the Phase 1-3 code and the TAWOS data, copied unchanged from my Software Engineering project (https://github.com/hani-ziad/SwEng_202610) so this repository runs on its own.
- `data/` has the derived tables, `results/` has all scores, intervals and the run log, and `report/` has the plan.
- `guide/` has `Rebuild_Manual.docx` (every step to rebuild the results by hand) and `Plain_Explanation.docx` (the project in plain words).

## What I reused

Phase 1 of the pipeline was my CSBP711 Assignment 1. Phases 2 and 3 (the history features and the in-sprint state rebuild) were built for my Software Engineering course project, and I've told both instructors I'm reusing them here. The burndown features and grades come from my CSBP711 paper critique. What's new in this project is the early-warning task itself, the dataset that brings the three feature sources together, the comparison with the paper's method and my Assignment 1 winner, and the analysis.

Data: TAWOS (Tawosi et al., MSR 2022), Apache 2.0.

## AI-use statement

I used Claude (Anthropic) as a coding and writing assistant. It helped write the code in `advai/`, ran the experiments, and drafted this README and the plan. I chose the question, the baselines and the evaluation, and reviewed all code and text. Every number comes from running this repository.
