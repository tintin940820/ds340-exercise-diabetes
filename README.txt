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

The raw CDC files are too big for GitHub. Download them here:
https://www.cdc.gov/brfss/annual_data/annual_2015.html
https://www.cdc.gov/brfss/annual_data/annual_2021.html
https://www.cdc.gov/brfss/annual_data/annual_2023.html

Files
prepare/ - makes the two clean tables from the raw files
tools.py - shared functions (load data, split, fit model, score, odds ratios, charts)
parent_paper_1.py - reproduces parent paper 1 (Aich et al. 2025) Table 3
q1_part1_which_exercise.py
q1_part2_exercise_vs_factors.py
q2_generations.py
results/ - output tables
figures/ - charts

Reproducing parent paper 1
Aich et al. (2025), A copula based supervised filter for feature selection in machine
learning driven diabetes risk prediction, arXiv 2505.22554.
Same CDC data (253,680 adults, 21 features), same 4 models (LR, RF, GB, XGBoost), all 21
features vs the Gumbel copula top 10, same setup as the paper (80/20 split, 20% of train
to pick the threshold, class weights, isotonic calibration).
Results: results/parent_paper_1_reproduction.csv (ours next to the paper's numbers)
and results/parent_paper_1_gumbel_scores.csv.

Test AUC, ours vs paper (Table 3):
              All 21 features     Gumbel top 10
LR            0.818 vs 0.820      0.814 vs 0.816
RF            0.807 vs 0.801      0.766 vs 0.741
GB            0.825 vs 0.827      0.821 vs 0.823
XGBoost       0.821 vs 0.827      0.817 vs 0.822
The Gumbel top 10 features match the paper exactly, in the same order.
The paper does not give tree settings or its random seed, so we used library defaults.

Method
Weighted logistic regression with the CDC survey weights.
Each script splits the data 70% train / 10% validation / 20% test (stratified, seed 42).
Models are fit on train, validation picks the model, test is used once for AUC.
Odds ratios come from the training set.

Run
pip install pandas numpy scipy statsmodels scikit-learn xgboost matplotlib pyreadstat
bash run_all.sh
python3 parent_paper_1.py   (takes about 10 minutes)

To rebuild the clean tables (inside prepare/):
python3 01_load_raw.py LLCP2015.XPT
python3 02_clean.py
python3 03_load_generations.py LLCP2015.XPT LLCP2021.XPT LLCP2023.XPT
python3 04_clean_generations.py
