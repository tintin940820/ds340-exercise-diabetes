DS 340 Group 27 - Which Exercise Protects Against Diabetes?
Tim Chen, Ying Jiang

The data and the code are in this same repository. The data is in the data folder.


QUESTIONS

Q1 part 1: Which exercise helps the most against diabetes?
Q1 part 2: How does exercise compare with weight, smoking, drinking and diet?
Q2: Do Gen Z (2023) and Millennial (2015) young adults, ages 18 to 29, differ in diabetes,
    and do their habits explain the difference?


DATA

CDC Behavioral Risk Factor Surveillance System (BRFSS). Public and de-identified.

data/analysis_core.csv.gz     BRFSS 2015, all adults, 253,680 people. Used for Q1.
data/generation_core.csv.gz   BRFSS 2015, 2021 and 2023, ages 18 to 29, 139,113 people. Used for Q2.

The outcome is 1 if the person was ever told they have diabetes or prediabetes.
Each row also has the CDC survey weight (weight).

The raw CDC files are over 1 GB, so they are not in this repo. They can be downloaded here:
2015: https://www.cdc.gov/brfss/annual_data/annual_2015.html
2021: https://www.cdc.gov/brfss/annual_data/annual_2021.html
2023: https://www.cdc.gov/brfss/annual_data/annual_2023.html


FILES

prepare/                           turns the raw CDC files into the two clean tables
q1_part1_which_exercise.py         Q1 part 1
q1_part2_exercise_vs_factors.py    Q1 part 2
q2_generations.py                  Q2
run_all.sh                         runs the three scripts
results/                           output tables and printouts
figures/                           one chart per question


METHOD

All three scripts do the same steps:

1. Load the clean table and the survey weight.
2. Turn each factor into a 0/1 column (1 = has the risk).
3. Split the data into 70% training, 10% validation and 20% test.
   The split keeps the same share of people with diabetes in each part (seed 42).
4. Look at the percent with diabetes in each group (training set).
5. Fit weighted logistic regression on the training set. The validation set picks
   the model that predicts new people best.
6. Check the chosen model once on the test set (AUC and log loss), then save the
   odds ratios, the model check table and a chart.

Odds ratio: 1 = no difference, below 1 = less diabetes, above 1 = more diabetes.

Q1 part 1: each person gets one label, their main sport or "No exercise". Every sport is
compared with no exercise. Only sports with at least 500 people are kept. BMI is left
out on purpose, because exercise partly works by lowering weight.

Q1 part 2: all factors go into one model together, so each odds ratio is that factor's
own effect with the others held equal.

Q2: we compare the same age group in 2015 and 2023, so the difference comes from the
generation and not from age. The Gen Z odds ratio is fit with age and sex first, then
obesity, exercise, smoking and drinking are added one at a time. If the odds ratio moves
toward 1 when a habit is added, that habit explains part of the gap.


HOW TO RUN

pip install pandas numpy statsmodels scikit-learn matplotlib pyreadstat
bash run_all.sh

To rebuild the clean tables from the raw CDC files:

cd prepare
python3 01_load_raw.py LLCP2015.XPT
python3 02_clean.py
python3 07_load_generations.py LLCP2015.XPT LLCP2021.XPT LLCP2023.XPT
python3 08_clean_generations.py


LIMITS

The survey is taken at one point in time, so these are associations, not proof of cause.
The 95% intervals use the survey weights but not the full survey design, so they are a
little too narrow.
