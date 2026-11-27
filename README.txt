DS 340 Group 27
Tim Chen, Ying Jiang

Data and code are in this same repo. The data is in the data folder.

Questions
Q1 part 1: which exercise helps the most against diabetes
Q1 part 2: how exercise compares with weight, smoking, drinking and diet
Q2: Gen Z (2023) vs Millennials (2015), ages 18-29

Data
CDC BRFSS survey (public).
data/analysis_core.csv.gz - 2015, all adults, 253,680 rows (Q1)
data/generation_core.csv.gz - 2015, 2021, 2023, ages 18-29, 139,113 rows (Q2)
Outcome = 1 if ever told they have diabetes or prediabetes.

Train, validation and test sets (data/splits/)
The whole project uses ONE split. make_splits.py makes it once, checks it, and saves it.
Every analysis script reads these files. No script makes its own split.
  train       70%  fit the models
  validation  10%  pick the model, the cutoff and the features
  test        20%  the unseen set. Each script opens it only at the very end, after
                   everything is picked, and uses it once (look for "opening the test set"
                   in the logs). We never went back and changed a model after seeing test.
Stratified on diabetes (Q2: on year and diabetes together), seed 42, so every set has the
same diabetes rate and anyone who reruns make_splits.py gets the same rows.
Each file has a "dataset" column (train / validation / test) and a "row_id" column
(row number in the clean table). make_splits.py stops with an error if any row is in two
sets, any row is missing, a set is not 70/10/20, or the diabetes rates differ.
  Q1 and parent paper 1 (2015, all adults)       Q2 (ages 18-29, 2015 and 2023)
  data/splits/q1_train.csv.gz       177,576       data/splits/q2_train.csv.gz       53,016
  data/splits/q1_validation.csv.gz   25,368       data/splits/q2_validation.csv.gz   7,574
  data/splits/q1_test.csv.gz         50,736       data/splits/q2_test.csv.gz        15,148
Diabetes rate: 15.8% in every Q1 set, 2.0% in every Q2 set.

How each script uses the three sets
  script                              train                validation               test
  q1_part1_which_exercise.py          fit 2 models         pick the better one      final AUC
  q1_part2_exercise_vs_factors.py     fit 2 models         pick the better one      final AUC
  q1_checks.py                        2 checks on Q1       -                        -
  q2_generations.py                   fit 4 models         pick the best one        final AUC
  parent_paper_1.py                   fit 8 models         pick each cutoff         compare with paper
  novelty.py                          fit 18 models,       pick each cutoff         compare feature sets
                                      pick features

The raw CDC files are too big for GitHub. Download them here:
https://www.cdc.gov/brfss/annual_data/annual_2015.html
https://www.cdc.gov/brfss/annual_data/annual_2021.html
https://www.cdc.gov/brfss/annual_data/annual_2023.html

Files
prepare/ - makes the two clean tables from the raw files
make_splits.py - step 1: makes and checks the train, validation and test sets (data/splits/)
run_all.py - runs every step in order
check_results.py - last step: compares the new results with the numbers in our paper
tools.py - shared functions (load data, split, fit model, score, odds ratios, charts)
parent_paper_1.py - reproduces parent paper 1 (Aich et al. 2025) Table 3
novelty.py - our change to parent paper 1 (new feature selection block), compared with other methods
q1_part1_which_exercise.py - Q1 part 1: odds of diabetes for each main sport vs no exercise
q1_part2_exercise_vs_factors.py - Q1 part 2: no exercise next to weight, smoking, drinking and diet
q1_checks.py - two checks on Q1 part 1 (from midterm feedback): without people in fair/poor
  health or with trouble walking, and minutes of activity per week instead of the main sport
q2_generations.py - Q2: Gen Z (2023) vs Millennials (2015) at ages 18-29
results/ - output tables
figures/ - charts

Reproducing parent paper 1
Aich et al. (2025), A copula based supervised filter for feature selection in machine
learning driven diabetes risk prediction, arXiv 2505.22554.
Same CDC data (253,680 adults, 21 features), same 4 models (LR, RF, GB, XGBoost), all 21
features vs the Gumbel copula top 10, same setup as the paper (class weights, isotonic
calibration, cutoff picked for the best F1). The paper uses an 80/20 split and holds back
part of the 80% to pick the cutoff. We use the project split: fit on train (70%), pick the
cutoff on validation (10%), score on test (20%, the same size as the paper's test set).
Results: results/parent_paper_1_reproduction.csv (ours next to the paper's numbers)
and results/parent_paper_1_gumbel_scores.csv.

Test AUC, ours vs paper (Table 3):
              All 21 features     Gumbel top 10
LR            0.817 vs 0.820      0.813 vs 0.816
RF            0.806 vs 0.801      0.760 vs 0.741
GB            0.825 vs 0.827      0.821 vs 0.823
XGBoost       0.822 vs 0.827      0.817 vs 0.822
The Gumbel top 10 features match the paper exactly, in the same order.
The paper does not give tree settings or its random seed, so we used library defaults.

Our change (novelty.py)
The block we changed is feature selection (select_features in tools.py).
Paper 1 scores each feature with the Gumbel copula, but any feature with tau <= 0 gets 0.
So features that lower diabetes risk (exercise, income, education, fruit, vegetables)
can never be picked. Our two-sided version flips those features first (the rotated
Gumbel copula), so they get a score from the size of tau: lambda = 2 - 2^(1 - |tau|).
Second change: swap the yes/no exercise answer for the exercise detail we recovered from
the raw file (minutes per week, MET of main activity, strength days per week, activity level).
We compare with two common filters: mutual information and L1 logistic regression.
Same data, split (data/splits/) and models as the reproduction. Results: results/novelty_comparison.csv,
results/novelty_two_sided_scores.csv, figures/novelty_comparison.png.

Test set results (protective = features that lower risk that were kept):
                                         protective  AUC LR/GB/XGB      F1 LR/GB/XGB
All 21 (paper)                               6       0.817/0.825/0.822  0.809/0.822/0.819
Gumbel top 10 (paper)                        0       0.813/0.821/0.817  0.803/0.811/0.812
Two-sided Gumbel top 10 (ours)               2       0.813/0.821/0.817  0.798/0.816/0.812
Mutual information top 10                    1       0.814/0.822/0.818  0.799/0.808/0.811
L1 logistic top 10                           2       0.816/0.824/0.821  0.806/0.812/0.809
Two-sided top 10 + exercise detail (ours)    4       0.813/0.821/0.817  0.801/0.814/0.817
Two-sided Gumbel keeps Income and PhysActivity in place of Stroke and CholCheck.
AUC is the same as the paper's top 10 for all 3 models, and F1 is about the same.
So we keep the same accuracy while keeping exercise and income in the model.
L1 logistic gives the best AUC with 10 features.

Main results (odds ratios from the training set, 95% interval)
Q1 part 1, vs no exercise: tennis 0.28, walking golf 0.30, running 0.31, basketball 0.31,
  yoga 0.31, weight lifting 0.33 ... walking 0.81, exercise bike 0.92 (0.82-1.02).
  results/q1_part1_which_exercise.csv, figures/q1_part1_which_exercise.png
Q1 part 2, all in one model: obese 4.42, overweight 2.02, no exercise 1.28, smoker 1.11,
  little vegetables 1.09, little fruit 1.08, heavy drinking 0.56.
  results/q1_part2_exercise_vs_factors.csv, figures/q1_part2_exercise_vs_factors.png
Q1 checks: without people in fair/poor health or with trouble walking, running 0.37,
  weight lifting 0.36, walking 0.92. By minutes per week vs 0: 1-149 min 0.82,
  150-299 min 0.78, 300+ min 0.61. results/q1_check_*.csv, figures/q1_check_healthier_only.png
Q2: diabetes or prediabetes 1.5% (2015) vs 2.8% (2023) at ages 18-29. Gen Z odds ratio
  1.83 (age, sex), 1.71 (+ obesity), 1.72 (+ exercise), 1.85 (+ smoking, drinking).
  results/q2_*.csv, figures/q2_generations.png

Method
Q1 and Q2: weighted logistic regression with the CDC survey weights.
Odds ratios come from the training set. The model is picked on validation (lowest log loss)
and checked once on test.

Run (Python 3.10 or newer)
1. Download the code: on GitHub click Code > Download ZIP and unzip it
   (or: git clone https://github.com/tintin940820/ds340-exercise-diabetes.git)
2. Open a terminal in the unzipped folder (Windows: Command Prompt or Anaconda Prompt)
3. pip install -r requirements.txt
4. python run_all.py       (Mac/Linux: python3 run_all.py; or bash run_all.sh)
run_all.py runs every script in order (about 20 minutes, a slower computer may take 30) and saves each printout to
results/log_<script>.txt. Tables go to results/ and charts to figures/.
The last step, check_results.py, prints each new number next to the number in our paper
(Tables III and IV and Section IV.D) and says "every number above matches the paper".
Or one at a time: python make_splits.py first, then any of the analysis scripts,
then python check_results.py.

To rebuild the clean tables (inside prepare/):
python3 01_load_raw.py LLCP2015.XPT
python3 02_clean.py
python3 03_load_generations.py LLCP2015.XPT LLCP2021.XPT LLCP2023.XPT
python3 04_clean_generations.py
