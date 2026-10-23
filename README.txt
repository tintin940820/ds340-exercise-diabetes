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
q1_part1_which_exercise.py
q1_part2_exercise_vs_factors.py
q2_generations.py
results/ - output tables
figures/ - charts

Method
Weighted logistic regression with the CDC survey weights.
Each script splits the data 70% train / 10% validation / 20% test (stratified, seed 42).
Models are fit on train, validation picks the model, test is used once for AUC.
Odds ratios come from the training set.

Run
pip install pandas numpy statsmodels scikit-learn matplotlib pyreadstat
bash run_all.sh

To rebuild the clean tables (inside prepare/):
python3 01_load_raw.py LLCP2015.XPT
python3 02_clean.py
python3 03_load_generations.py LLCP2015.XPT LLCP2021.XPT LLCP2023.XPT
python3 04_clean_generations.py
