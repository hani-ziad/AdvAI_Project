# Plan (written before any result of this project was run)

1 Nov 2026. Option 1 of the CSBP711 final project: close the gap I found in my paper critique. (The wording was tidied later the same day; the content is unchanged.)

## The gap

Pérez Castillo et al. (Information, 2024) grade a sprint as Poor, Fair, Good or Excellent from 11 burndown features (R1 to R11) taken at the end of the sprint. They pitch it as a way to catch sprints that need "immediate intervention", but they never test it before the sprint is over. In my critique (github.com/hani-ziad/AdvAI_critique), the same features on TAWOS lost most of their accuracy when only the first 25% of the sprint was visible (.77 down to .51), and one feature, "% done", carried almost all of the signal.

## Research question

A quarter or halfway into a sprint, can we tell which sprints will end Poor? And does adding what Jira already knows (the team's recent history and the sprint's scope changes, from my Assignment 1 and Software Engineering pipeline) beat the paper's burndown-only features?

## Data and target

- TAWOS, 8 open-source Jira projects, the 1,691 sprints with committed work (the same group as Assignment 1).
- Target: whether the sprint ends Poor, using the grade from the critique (main grade: half pace, half final completion, cut at the paper's class sizes on the training sprints). As a check, the "pace only" grade.
- At each checkpoint I keep only the sprints still open at that moment, as in Phase 3.

## Models (all trained on the same data)

1. The base rate and "% done so far" (no learning: the less that's done, the higher the risk).
2. The paper's method: SVM on R1 to R11 computed from the days seen so far, trained on the 4 grades as in the paper, with its Poor probability as the score.
3. My Assignment 1 winner: the Phase 1 random forest (300 trees, depth 5, leaf 3, balanced) on its planning-time features plus project, retrained for this target.
4. Mine: gradient boosting (Phase 2 settings) on R1 to R11 + Phase 3 sprint-state features + Phase 2 history features + project, with a random forest on the same features as a check.

## Evaluation

- Main split: the Assignment 1 time split (the first 70% of each project's sprints for training, the last 30% for testing), seed 42. Second split: leave one project out.
- Main metric: PR-AUC for Poor. It copes with the class balance and reads as "how clean are the alarms". I also report ROC-AUC and the precision when 80% of Poor sprints are caught.
- Uncertainty: a bootstrap over the 8 projects in the test set (2,000 resamples), for each model and for the difference between mine and the paper's method.
- Secondary: 4-grade macro F1 for the paper's SVM, the Assignment 1 RF and mine.

## What counts as an answer

Mine beats the paper's method at a checkpoint if the bootstrap interval of the PR-AUC difference sits above zero. If it doesn't, I report that. Later, if there's time: the MariaDB outside check, and Shapley explanations of what triggers an alarm.

## Change after the first run (1 Nov 2026)

After the first run I noticed that "mine against the paper's method" changes two things at once, the algorithm and the features. So I added one control model, gradient boosting on R1 to R11 only, to show the gain from the extra features on its own. Nothing else in the plan changed, and the results of the original five models stayed the same.
