# AdvAI_Project: spotting Poor sprints early

My final project for CSBP711 Advanced AI, Fall 2026 (Option 1: close the gap from the paper critique).
Hani AbuSharkh, HCT Al Ain.

## The question

A quarter or halfway into a sprint, can we tell which sprints are going to end Poor? And does using what Jira already knows (the team's recent history, plus the sprint's current state and scope changes) do better than the burndown-only features of Pérez Castillo et al. (*Information*, 2024)?

Here is the gap I'm closing. The paper grades a sprint from 11 burndown features (R1 to R11) taken at the end of the sprint, but sells the method as a way to catch sprints that need "immediate intervention". In my critique (https://github.com/hani-ziad/AdvAI_critique), those same features on TAWOS dropped from .77 accuracy at the end of the sprint to .51 at the 25% mark. So this project builds a real early warning and tests it.

I wrote the plan before running anything (`report/plan.md`). After the first run I added one control model, gradient boosting on R1 to R11 only, and the plan explains why.

## Setup

- **Data:** TAWOS, 8 open-source Jira projects, the 1,691 sprints with committed work (the same group as Assignment 1). At each checkpoint I only keep sprints that are still open: 1,663 at 25% and 1,646 at 50%.
- **Target:** whether the sprint ends Poor (about 27% of sprints), using the grade from the critique. A "pace only" grade is used as a check.
- **Splits:** the Assignment 1 time split (the first 70% of each project's sprints for training, the last 30% for testing), and leaving one project out. Seed 42.
- **Uncertainty:** a bootstrap over the 8 projects, 2,000 resamples.

| Model | Features | Why it's here |
|---|---|---|
| % done so far | R1 only, no learning | the simplest possible check |
| Paper: SVM on R1-R11 | the paper's 11 burndown features, trained on the 4 grades | the critique paper's method |
| GB on R1-R11 only | the paper's features, my algorithm | separates the algorithm from the features |
| Assignment 1 winner | the Phase 1 random forest on planning-time features + project | my Assignment 1 winner |
| **Mine: GB, all features** | R1-R11 + Phase 3 sprint state + Phase 2 history (37 numeric features) + project | the proposed model |
| Mine: RF, all features | same | a second check |

## Results

PR-AUC for "ends Poor" with the main grade. Random guessing gets about .31 on the time-split test set.

| Model | 25%, time split | 25%, across projects | 50%, time split | 50%, across projects |
|---|---|---|---|---|
| % done so far | .533 | .515 | .656 | .684 |
| Paper: SVM on R1-R11 | .525 | .414 | .666 | .579 |
| GB on R1-R11 only | .589 | .463 | .700 | .628 |
| Assignment 1 winner (RF) | .589 | .485 | .595 | .477 |
| **Mine: GB, all features** | **.714** | **.662** | **.777** | **.781** |
| Mine: RF, all features | .720 | .720 | .780 | .804 |

How much mine beats the paper's method (PR-AUC difference, 95% project-bootstrap interval):

| | 25% | 50% |
|---|---|---|
| Time split | +.189 [.064, .270] | +.111 [−.009, .178] |
| Across projects | +.248 [.073, .308] | +.202 [.058, .243] |

What this tells me:

- **Early in the sprint, the Jira context is what helps.** At 25%, my model beats the paper's method, the same algorithm on the paper's features (+.125 [.018, .192]), my Assignment 1 winner and "% done". The burndown features on their own add little over "% done", and the paper's SVM doesn't even beat it.
- **By the midpoint the burndown catches up.** At 50% on the time split, the gain over the paper is +.11, but the interval just touches zero. Across projects it stays clear (+.20).
- **The alarms are cleaner.** If the model has to catch 80% of Poor sprints, about 60% of my alarms are right at 25% and 71% at 50%, against 55% and 65% for the paper's method.
- **Planning-time information isn't enough on its own.** The Assignment 1 winner doesn't improve between 25% and 50%, because it never sees how the sprint is going.
- **The pace-only grade tells the same story.** All 16 of its comparisons have intervals above zero (see `results/early_warning_differences.csv`).

## How to run

```
pip install -r requirements.txt
python -m advai.burndown          # about 15 s, writes data/burndown_daily.csv
python -m advai.build_dataset     # about 20 s, writes data/early_warning_frame.csv
python -m advai.run_experiments   # about 4 min, writes results/
```

Run these from the repository folder. My run used Python 3.13, scikit-learn 1.9.1, pandas 3.0 and numpy 2.5 (`results/summary.json`).

## Files

- `advai/` is the project code: the burndown rebuild and the R1-R11 features and grades (both copied from the critique), the dataset builder and the experiments.
- `pipeline/` holds the Phase 1-3 code and the TAWOS data, copied unchanged from my Software Engineering project (https://github.com/hani-ziad/SwEng_202610) so this repository runs on its own.
- `data/` has the derived tables, `results/` has all scores, intervals and the run log, and `report/` has the plan.

## What I reused

Phase 1 of the pipeline was my CSBP711 Assignment 1. Phases 2 and 3 (the history features and the in-sprint state rebuild) were built for my Software Engineering course project, and I've told both instructors I'm reusing them here. The burndown features and grades come from my CSBP711 paper critique. What's new in this project is the early-warning task itself, the dataset that brings the three feature sources together, the comparison with the paper's method and my Assignment 1 winner, and the analysis.

Data: TAWOS (Tawosi et al., MSR 2022), Apache 2.0.

## AI-use statement

I used Claude (Anthropic) as a coding and writing assistant. It helped write the code in `advai/`, ran the experiments, and drafted this README and the plan. I chose the question, the baselines and the evaluation, and reviewed all code and text. Every number comes from running this repository.
