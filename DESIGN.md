# DS340: Code Design

## The questions
- **Q1 part 1.** Which exercise helps the most against diabetes?
- **Q1 part 2.** How does exercise compare with other factors (weight, smoking, drinking, diet)?
- **Q2.** How do Gen Z and Millennial young adults compare (habits, diabetes, and whether habits explain the difference)?

## Folder layout
```
prepare/                          reads the raw CDC files once and writes the two clean tables (already run)
data/analysis_core.csv.gz         2015, all adults (253,680)       -> used by Q1
data/generation_core.csv.gz       2015/2021/2023, ages 18 to 29    -> used by Q2
q1_part1_which_exercise.py        Q1 part 1
q1_part2_exercise_vs_factors.py   Q1 part 2
q2_generations.py                 Q2
run_all.sh                        runs the three scripts
results/                          tables (.csv) and printouts (log_*.txt)
figures/                          one chart per question
```

## The design in one sentence
**There is one script per question, and every script follows the same six steps:**

| Step | What it does | Why |
|---|---|---|
| 1. Load | Reads the clean table and adds the CDC survey weight | The weight makes the answer describe all US adults, not just the people who happened to answer |
| 2. Make 0/1 columns | Turns each thing we compare into a yes/no column (or one label, such as "main sport") | One coding rule makes every odds ratio mean the same thing |
| 3. Split 70 / 10 / 20 | 70% training, 10% validation, 20% test, with the same share of people with diabetes in each part | The model is fit on one part of the data and checked on people it has never seen |
| 4. Simple look | Percent with diabetes in each group, on the training set, with no model | A plain number anyone can check, and a baseline to compare the model against |
| 5. Train and validate | Fits weighted logistic regression models on the training set; the validation set picks the model that fits new people best | Exercisers are younger, richer and smoke less, so the model holds those things equal; the validation set chooses between model versions without touching the test set |
| 6. Test and save | Checks the chosen model once on the test set (AUC and log loss), then saves the odds ratio table, the model check table and one chart | The test set gives an honest score, because nothing was tuned on it |

## Why this design fits our questions

**Why one script per question.** Each question uses different data and asks something different. Keeping them apart means:
- you can run or read one question without the others
- a change to one question cannot break another
- when you present, each file is one slide's worth of work

**Why the same six steps everywhere.** Once you understand one script you understand all three. The only parts that change between questions are **step 2 (what we compare)** and **step 5 (which models we try)**.

**Why a 70 / 10 / 20 split.**
- **Training (70%):** every model is fit here, and every odds ratio we report comes from here.
- **Validation (10%):** each script fits more than one version of the model. The validation set picks the version with the lowest log loss, which means it predicts new people best. Because this choice is made on the validation set, the test set stays untouched.
- **Test (20%):** used once, at the very end, for an honest score of the chosen model (AUC and log loss).
- **The split is stratified.** Each part has the same share of people with diabetes (and for Q2, the same mix of 2015 and 2023). This matters because diabetes is rare in young adults (about 2%), and a random split could leave too few cases in a small part.
- **It uses a fixed seed (42),** so anyone who runs the code gets the same split.
- **We report AUC and log loss, not accuracy.** Diabetes is uncommon, so a model that always says "no diabetes" would get high accuracy while being useless.

**Why logistic regression.** The outcome is yes or no (diabetes or prediabetes). Logistic regression is the standard model for a yes/no outcome. It gives an **odds ratio**, which reads simply:
- 1 means no difference
- below 1 means less diabetes
- above 1 means more diabetes

It is also what our parent papers used, so our results can be compared with theirs.

**Why survey weights.** BRFSS deliberately surveys some groups more than others. Without the weight the percentages would be off. We divide the weight by its average so it only changes how much each person counts, not the size of the sample. The validation and test scores are weighted too.

### Q1 part 1: which exercise helps most
- **Step 2:** each person gets one label, either their main sport or "No exercise". We keep only sports with at least 500 people, because a sport with 40 people would give a noisy result that could top the list by luck.
- **Step 5:** two candidate models, both with "sport" as one variable and **"No exercise" as the reference group**. The basic model holds age and sex equal. The full model also holds income, education, smoking, drinking and diet equal. The validation set chooses between them. Every sport is compared with the same baseline, so the odds ratios can be ranked directly.
- **What we hold equal and why:** these are the things that differ between, say, runners and walkers and that also affect diabetes. Without them, running would look good simply because runners are young.
- **What we leave out on purpose:** body weight (BMI). Exercise partly works *through* weight. Holding weight equal would hide part of the benefit we are trying to measure.

### Q1 part 2: exercise vs other factors
- **Step 2:** every factor is coded as **1 = has the risk** (obese, no exercise, smoker, little fruit, and so on). That is why exercise appears as "No_exercise". Coded this way, every odds ratio points the same direction and you can rank them by size.
- **Step 5:** all the factors go into **one model together**. If each factor got its own model, obesity would get credit for some of the effect of not exercising, and the other way round. In one model, each factor's number is its own effect with the others held equal. The two candidates are the factors alone, and the factors plus age, sex, income and education; the validation set chooses.
- Here BMI *is* in the model, because weight is one of the factors we are comparing.

### Q2: generations
- **Why compare 2015 with 2023, ages 18 to 29.** Young adults in 2015 were Millennials and young adults in 2023 are Gen Z. Because both groups are the same age, any difference comes from the generation, not from getting older.
- **Step 1:** we keep only people who answered every question used here, so every model in step 5 uses exactly the same people. Otherwise a change in the odds ratio could come from a change in who is counted.
- **Step 5 is the key design choice.** We fit the Gen Z odds ratio once with only age and sex, then **add habits one step at a time**:
  - if the odds ratio moves toward 1 when a habit is added, that habit explains part of the gap
  - if it doesn't move, it doesn't explain the gap

  All four models are fit on the training set, and the validation set shows which one predicts new people best.

## What the results say
These numbers come from the training set (odds ratios) and the test set (AUC).

- **Q1 part 1.** Every exercise beats no exercise. Tennis (0.29), walking golf (0.31), running (0.33), weight lifting and basketball (0.34) are linked to about two thirds lower odds. Walking is the most common exercise but among the weakest (0.82). Exercise bike (0.97) and stair climbing (0.91) barely differ from no exercise. The validation set chose the full model, and its test AUC is 0.74.
- **Q1 part 2.** Weight matters most: obese 4.42 and overweight 2.02. Not exercising comes next at 1.28, which is larger than smoking (1.11) or diet (about 1.08 to 1.09). Heavy drinking shows 0.56. This is a known quirk of survey data (heavy drinkers who are sick tend to stop drinking), so it should not be read as drinking being protective. The validation set chose the model with age, sex, income and education, and its test AUC is 0.78.
- **Q2.** Gen Z young adults have more diabetes or prediabetes: 2.8% vs 1.5% in the training set, odds ratio 1.83.
  - **Habits changed in both directions:** obesity is up about 4 points, while smoking is down about 11 points and binge drinking is down about 3 points.
  - **What explains the gap:** holding obesity equal shrinks the odds ratio from 1.83 to 1.71, so rising obesity explains part of it. Adding smoking and drinking pushes it back up to 1.85, because Gen Z smokes far less, which should have *lowered* their risk.
  - **Validation:** the model with obesity predicts new people best, and adding exercise, smoking or drinking does not improve it. Its test AUC is 0.69.
  - **Bottom line:** habits as a whole do not explain why Gen Z has more diabetes. Obesity explains some of it, and the rest is something these questions do not measure (for example more testing, other diet changes, or sleep).

## Limits to mention in the paper
- **The data is a survey taken at one point in time.** These are associations, not proof that a sport causes lower diabetes. For example, people who are already healthy may be more likely to choose running.
- **The 95% intervals are a little too narrow.** They use the weights but not the survey's cluster design. The conclusions don't change.
- **Odds ratios use 70% of the data.** That makes the intervals slightly wider than with all the data, in exchange for an honest test score.
- **Main sport only.** "Main sport" is the one activity each person named first.

## All the code

### run_all.sh
```bash
#!/bin/bash
# Runs the three questions in order (about 1 minute). Tables go to results/, charts to figures/.
cd "$(dirname "$0")"
for q in q1_part1_which_exercise q1_part2_exercise_vs_factors q2_generations; do
  python3 -W ignore $q.py | tee results/log_$q.txt
done
```

### q1_part1_which_exercise.py
```python
"""Question 1, part 1: Which exercise helps the most against diabetes?
Data: CDC BRFSS 2015, 253,680 US adults. Compares each person's main exercise with doing no exercise.
Split: 70% training (fit), 10% validation (choose the model), 20% test (final check, used once)."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# STEP 1. Load the clean data and the survey weight
d = pd.read_csv("data/analysis_core.csv.gz")
d["w"] = d.weight / d.weight.mean()   # CDC weight, so results speak for all US adults, not just who answered

# STEP 2. Give each person one label: their main exercise, or "No exercise"
names = {2: "Aerobics", 5: "Basketball", 6: "Exercise bike", 7: "Bicycling", 11: "Calisthenics", 14: "Dancing",
         15: "Elliptical", 18: "Gardening", 19: "Golf (cart)", 20: "Golf (walking)", 22: "Hiking", 28: "Jogging",
         31: "Mowing lawn", 37: "Running", 54: "Stair climbing", 57: "Swimming", 58: "Swimming laps",
         61: "Tennis", 64: "Walking", 67: "Weight lifting", 69: "Yoga", 73: "Household chores", 76: "Yard work"}
d["sport"] = d.MainSport.map(names)
d.loc[d.MainSport == 0, "sport"] = "No exercise"
d = d.dropna(subset=["sport"])
counts = d.sport.value_counts()
d = d[d.sport.map(counts) >= 500]     # keep sports with at least 500 people, so each result is reliable
print(f"STEP 2  {len(d)} people, {d.sport.nunique() - 1} sports (plus 'No exercise')")

# STEP 3. Split 70% training, 10% validation, 20% test (same share with diabetes in each part)
train, rest = train_test_split(d, test_size=0.30, stratify=d.Diabetes, random_state=42)
valid, test = train_test_split(rest, test_size=2 / 3, stratify=rest.Diabetes, random_state=42)
print(f"STEP 3  training {len(train)}, validation {len(valid)}, test {len(test)}")

# STEP 4. Simple look (training set): percent with diabetes in each group
raw = train.groupby("sport").apply(lambda g: 100 * np.average(g.Diabetes, weights=g.w))
print("\nSTEP 4  percent with diabetes or prediabetes, training set, before any adjustment:")
print(raw.sort_values().round(1).to_string())

# STEP 5. Fit two candidate models on the training set; the validation set picks the one that fits new people best
SPORT = "C(sport, Treatment('No exercise'))"
candidates = {"basic (age, sex)": f"Diabetes ~ {SPORT} + C(Age) + Sex",
              "full (+ income, education, smoking, drinking, diet)":
                  f"Diabetes ~ {SPORT} + C(Age) + Sex + C(Income) + C(Education) + Smoker + HvyAlcoholConsump + Fruits + Veggies"}
fits, checks = {}, []
for name, formula in candidates.items():
    fits[name] = smf.glm(formula, data=train, family=sm.families.Binomial(), var_weights=train.w).fit()
    p = fits[name].predict(valid)
    checks.append({"model": name, "split": "validation", "log_loss": log_loss(valid.Diabetes, p, sample_weight=valid.w),
                   "AUC": roc_auc_score(valid.Diabetes, p, sample_weight=valid.w)})
best = min(checks, key=lambda c: c["log_loss"])["model"]
print("\nSTEP 5  validation check (lower log loss = better fit to new people):")
print(pd.DataFrame(checks).round(4).to_string(index=False))
print(f"        chosen model: {best}")

# STEP 6. Test set, used once: how well does the chosen model work on people it has never seen?
model = fits[best]
p = model.predict(test)
checks.append({"model": best, "split": "test", "log_loss": log_loss(test.Diabetes, p, sample_weight=test.w),
               "AUC": roc_auc_score(test.Diabetes, p, sample_weight=test.w)})
print(f"\nSTEP 6  test set: AUC = {checks[-1]['AUC']:.3f}, log loss = {checks[-1]['log_loss']:.4f}")

# Odds ratios from the chosen model (fit on the training set)
rows = [k for k in model.params.index if k.startswith("C(sport")]
table = pd.DataFrame({"odds_ratio": np.exp(model.params[rows]),
                      "low_95": np.exp(model.conf_int().loc[rows, 0]),
                      "high_95": np.exp(model.conf_int().loc[rows, 1])})
table.index = [k.split("T.")[1].rstrip("]") for k in rows]
table["people_train"] = train.sport.value_counts()[table.index]
table = table.sort_values("odds_ratio")
print("\n        odds ratio vs no exercise (1 = same as no exercise, lower = more protective):")
print(table.round(2).to_string())

# Save the tables and a chart
table.to_csv("results/q1_part1_which_exercise.csv")
pd.DataFrame(checks).to_csv("results/q1_part1_model_check.csv", index=False)
fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.barh(table.index[::-1], table.odds_ratio[::-1], color="#1f63a8")
ax.axvline(1, color="#c4622d")
ax.set_xlabel("Odds of diabetes compared with no exercise (lower = more protective)")
fig.savefig("figures/q1_part1_which_exercise.png", dpi=200, bbox_inches="tight")
```

### q1_part2_exercise_vs_factors.py
```python
"""Question 1, part 2: How does exercise compare with other factors (weight, smoking, drinking, diet)?
Data: CDC BRFSS 2015, 253,680 US adults. All factors go into one model so they can be ranked side by side.
Split: 70% training (fit), 10% validation (choose the model), 20% test (final check, used once)."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# STEP 1. Load the clean data and the survey weight
d = pd.read_csv("data/analysis_core.csv.gz")
d["w"] = d.weight / d.weight.mean()

# STEP 2. Turn every factor into a 0/1 "risk" column (1 = has the risk), so all odds ratios read the same way
d["Obese"] = (d.BMI >= 30).astype(int)
d["Overweight"] = ((d.BMI >= 25) & (d.BMI < 30)).astype(int)
d["No_exercise"] = 1 - d.PhysActivity
d["Smoker"] = d.Smoker
d["Heavy_drinking"] = d.HvyAlcoholConsump
d["Little_fruit"] = 1 - d.Fruits
d["Little_vegetables"] = 1 - d.Veggies
factors = ["Obese", "Overweight", "No_exercise", "Smoker", "Heavy_drinking", "Little_fruit", "Little_vegetables"]

# STEP 3. Split 70% training, 10% validation, 20% test (same share with diabetes in each part)
train, rest = train_test_split(d, test_size=0.30, stratify=d.Diabetes, random_state=42)
valid, test = train_test_split(rest, test_size=2 / 3, stratify=rest.Diabetes, random_state=42)
print(f"STEP 3  training {len(train)}, validation {len(valid)}, test {len(test)}")

# STEP 4. Simple look (training set): percent with diabetes with and without each factor
print("\nSTEP 4  percent with diabetes or prediabetes, training set, before any adjustment:")
for f in factors:
    has = 100 * np.average(train.Diabetes[train[f] == 1], weights=train.w[train[f] == 1])
    no = 100 * np.average(train.Diabetes[train[f] == 0], weights=train.w[train[f] == 0])
    print(f"        {f:18s} has it: {has:5.1f}%   does not: {no:5.1f}%")

# STEP 5. Fit two candidate models on the training set; the validation set picks the one that fits new people best
FACT = " + ".join(factors)
candidates = {"factors only": f"Diabetes ~ {FACT}",
              "factors + age, sex, income, education": f"Diabetes ~ {FACT} + C(Age) + Sex + C(Income) + C(Education)"}
fits, checks = {}, []
for name, formula in candidates.items():
    fits[name] = smf.glm(formula, data=train, family=sm.families.Binomial(), var_weights=train.w).fit()
    p = fits[name].predict(valid)
    checks.append({"model": name, "split": "validation", "log_loss": log_loss(valid.Diabetes, p, sample_weight=valid.w),
                   "AUC": roc_auc_score(valid.Diabetes, p, sample_weight=valid.w)})
best = min(checks, key=lambda c: c["log_loss"])["model"]
print("\nSTEP 5  validation check (lower log loss = better fit to new people):")
print(pd.DataFrame(checks).round(4).to_string(index=False))
print(f"        chosen model: {best}")

# STEP 6. Test set, used once: how well does the chosen model work on people it has never seen?
model = fits[best]
p = model.predict(test)
checks.append({"model": best, "split": "test", "log_loss": log_loss(test.Diabetes, p, sample_weight=test.w),
               "AUC": roc_auc_score(test.Diabetes, p, sample_weight=test.w)})
print(f"\nSTEP 6  test set: AUC = {checks[-1]['AUC']:.3f}, log loss = {checks[-1]['log_loss']:.4f}")

# Odds ratios from the chosen model (fit on the training set)
table = pd.DataFrame({"odds_ratio": np.exp(model.params[factors]),
                      "low_95": np.exp(model.conf_int().loc[factors, 0]),
                      "high_95": np.exp(model.conf_int().loc[factors, 1])}).sort_values("odds_ratio", ascending=False)
print("\n        odds ratio for each factor, all in one model (above 1 = more diabetes, bigger = stronger):")
print(table.round(2).to_string())

# Save the tables and a chart (exercise highlighted)
table.to_csv("results/q1_part2_exercise_vs_factors.csv")
pd.DataFrame(checks).to_csv("results/q1_part2_model_check.csv", index=False)
fig, ax = plt.subplots(figsize=(6.5, 3.5))
colors = ["#c4622d" if f == "No_exercise" else "#1f63a8" for f in table.index[::-1]]
ax.barh(table.index[::-1], table.odds_ratio[::-1], color=colors)
ax.axvline(1, color="#222222")
ax.set_xlabel("Odds ratio for diabetes (above 1 = more diabetes)")
fig.savefig("figures/q1_part2_exercise_vs_factors.png", dpi=200, bbox_inches="tight")
```

### q2_generations.py
```python
"""Question 2: Gen Z vs Millennial young adults (ages 18 to 29).
Millennials = young adults surveyed in 2015. Gen Z = young adults surveyed in 2023. Same age, different generation.
How do their habits differ, who has more diabetes, and do the habits explain the difference?
Split: 70% training (fit), 10% validation (compare the models), 20% test (final check, used once)."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# STEP 1. Load the clean data, keep 2015 and 2023, and the survey weight
d = pd.read_csv("data/generation_core.csv.gz")
d = d[d.year.isin([2015, 2023])]
d = d.dropna(subset=["diabetes", "weight_status", "inactive", "current_smoker", "binge"])  # same people in every step
d["w"] = d.weight / d.weight.mean()
print("STEP 1  young adults:", d.groupby("year").size().to_dict())

# STEP 2. 0/1 columns: generation and each habit
d["GenZ"] = (d.year == 2023).astype(int)
d["Obese"] = (d.weight_status == "Obese").astype(int)
d["No_exercise"] = d.inactive
d["Smoker"] = d.current_smoker
d["Binge_drinking"] = d.binge
habits = ["Obese", "No_exercise", "Smoker", "Binge_drinking"]

# STEP 3. Split 70% training, 10% validation, 20% test (same year mix and same share with diabetes in each part)
group = d.year.astype(str) + "_" + d.diabetes.astype(int).astype(str)
train, rest = train_test_split(d, test_size=0.30, stratify=group, random_state=42)
valid, test = train_test_split(rest, test_size=2 / 3, stratify=group[rest.index], random_state=42)
print(f"STEP 3  training {len(train)}, validation {len(valid)}, test {len(test)}")

# STEP 4. Simple look (training set): percent with each habit, and with diabetes, in each generation
rows = {}
for col in habits + ["diabetes"]:
    rows[col] = {gen: 100 * np.average(g[col], weights=g.w) for gen, g in
                 (("Millennial_2015", train[train.GenZ == 0]), ("GenZ_2023", train[train.GenZ == 1]))}
pct = pd.DataFrame(rows).T
pct["change"] = pct.GenZ_2023 - pct.Millennial_2015
print("\nSTEP 4  percent of young adults, training set:")
print(pct.round(1).to_string())

# STEP 5. Fit the Gen Z model four times on the training set, adding habits one step at a time.
# If the Gen Z odds ratio moves toward 1 when a habit is added, that habit explains part of the gap.
# The validation set checks which of the four models fits new people best.
steps = {"same age and sex": [],
         "+ same obesity": ["Obese"],
         "+ same exercise": ["Obese", "No_exercise"],
         "+ same smoking and drinking": ["Obese", "No_exercise", "Smoker", "Binge_drinking"]}
fits, out, checks = {}, [], []
for name, extra in steps.items():
    m = smf.glm("diabetes ~ GenZ + male + age_25_29" + "".join(" + " + h for h in extra),
                data=train, family=sm.families.Binomial(), var_weights=train.w).fit()
    fits[name] = m
    ci = np.exp(m.conf_int().loc["GenZ"])
    out.append({"model": name, "GenZ_odds_ratio": np.exp(m.params["GenZ"]), "low_95": ci[0], "high_95": ci[1]})
    p = m.predict(valid)
    checks.append({"model": name, "split": "validation", "log_loss": log_loss(valid.diabetes, p, sample_weight=valid.w),
                   "AUC": roc_auc_score(valid.diabetes, p, sample_weight=valid.w)})
table = pd.DataFrame(out).set_index("model")
print("\nSTEP 5  Gen Z odds of diabetes compared with Millennials, training set (1 = no difference):")
print(table.round(2).to_string())
best = min(checks, key=lambda c: c["log_loss"])["model"]
print("\n        validation check (lower log loss = better fit to new people):")
print(pd.DataFrame(checks).round(4).to_string(index=False))
print(f"        best-fitting model: {best}")

# STEP 6. Test set, used once: how well does the best model work on people it has never seen?
p = fits[best].predict(test)
checks.append({"model": best, "split": "test", "log_loss": log_loss(test.diabetes, p, sample_weight=test.w),
               "AUC": roc_auc_score(test.diabetes, p, sample_weight=test.w)})
print(f"\nSTEP 6  test set: AUC = {checks[-1]['AUC']:.3f}, log loss = {checks[-1]['log_loss']:.4f}")

# Save the tables and a chart
pct.to_csv("results/q2_habits_and_diabetes.csv")
table.to_csv("results/q2_genz_odds_ratio.csv")
pd.DataFrame(checks).to_csv("results/q2_model_check.csv", index=False)
fig, ax = plt.subplots(figsize=(6.5, 3.5))
y = np.arange(len(pct))
ax.barh(y - 0.2, pct.Millennial_2015, height=0.4, color="#9a9a9a", label="Millennials (2015)")
ax.barh(y + 0.2, pct.GenZ_2023, height=0.4, color="#1f63a8", label="Gen Z (2023)")
ax.set_yticks(y)
ax.set_yticklabels(pct.index)
ax.invert_yaxis()
ax.set_xlabel("Percent of young adults aged 18 to 29 (training set)")
ax.legend(frameon=False)
fig.savefig("figures/q2_generations.png", dpi=200, bbox_inches="tight")
```

## What the code printed

### results/log_q1_part1_which_exercise.txt
```
STEP 2  238404 people, 23 sports (plus 'No exercise')
STEP 3  training 166882, validation 23840, test 47682

STEP 4  percent with diabetes or prediabetes, training set, before any adjustment:
sport
Running              2.7
Basketball           3.5
Weight lifting       3.7
Tennis               4.9
Yoga                 5.7
Jogging              6.0
Golf (walking)       6.2
Hiking               7.1
Elliptical           7.4
Aerobics             8.4
Bicycling            8.6
Swimming laps        8.7
Calisthenics         9.8
Dancing             10.5
Swimming            12.2
Gardening           12.8
Golf (cart)         13.9
Walking             15.6
Mowing lawn         16.3
Yard work           17.5
Exercise bike       17.9
Household chores    18.6
Stair climbing      19.8
No exercise         20.9

STEP 5  validation check (lower log loss = better fit to new people):
                                              model      split  log_loss    AUC
                                   basic (age, sex) validation    0.3629 0.7400
full (+ income, education, smoking, drinking, diet) validation    0.3541 0.7627
        chosen model: full (+ income, education, smoking, drinking, diet)

STEP 6  test set: AUC = 0.743, log loss = 0.3631

        odds ratio vs no exercise (1 = same as no exercise, lower = more protective):
                  odds_ratio  low_95  high_95  people_train
Tennis                  0.29    0.20     0.44           563
Golf (walking)          0.31    0.23     0.42           839
Running                 0.33    0.30     0.38          8238
Weight lifting          0.34    0.30     0.38          4762
Basketball              0.34    0.24     0.48           625
Yoga                    0.45    0.36     0.57          1340
Bicycling               0.54    0.48     0.60          3537
Gardening               0.54    0.50     0.58          7663
Elliptical              0.54    0.46     0.63          2534
Jogging                 0.55    0.45     0.67          1165
Swimming laps           0.56    0.43     0.74           750
Hiking                  0.58    0.47     0.73          1227
Mowing lawn             0.59    0.46     0.75           561
Aerobics                0.59    0.51     0.68          3347
Calisthenics            0.61    0.53     0.70          2175
Golf (cart)             0.63    0.55     0.73          1869
Dancing                 0.66    0.54     0.81           846
Swimming                0.74    0.62     0.88          1279
Yard work               0.77    0.67     0.87          2253
Household chores        0.78    0.63     0.97           791
Walking                 0.82    0.80     0.85         73734
Stair climbing          0.91    0.70     1.18           426
Exercise bike           0.97    0.87     1.08          3244
```

### results/log_q1_part2_exercise_vs_factors.txt
```
STEP 3  training 177576, validation 25368, test 50736

STEP 4  percent with diabetes or prediabetes, training set, before any adjustment:
        Obese              has it:  23.6%   does not:   9.5%
        Overweight         has it:  12.5%   does not:  14.8%
        No_exercise        has it:  20.9%   does not:  11.7%
        Smoker             has it:  16.7%   does not:  11.8%
        Heavy_drinking     has it:   7.1%   does not:  14.4%
        Little_fruit       has it:  15.6%   does not:  12.9%
        Little_vegetables  has it:  17.6%   does not:  13.0%

STEP 5  validation check (lower log loss = better fit to new people):
                                model      split  log_loss    AUC
                         factors only validation    0.3981 0.6789
factors + age, sex, income, education validation    0.3581 0.7781
        chosen model: factors + age, sex, income, education

STEP 6  test set: AUC = 0.782, log loss = 0.3429

        odds ratio for each factor, all in one model (above 1 = more diabetes, bigger = stronger):
                   odds_ratio  low_95  high_95
Obese                    4.42    4.24     4.61
Overweight               2.02    1.93     2.11
No_exercise              1.28    1.24     1.32
Smoker                   1.11    1.08     1.15
Little_vegetables        1.09    1.06     1.13
Little_fruit             1.08    1.05     1.12
Heavy_drinking           0.56    0.52     0.60
```

### results/log_q2_generations.txt
```
STEP 1  young adults: {2015: 35367, 2023: 40371}
STEP 3  training 53016, validation 7574, test 15148

STEP 4  percent of young adults, training set:
                Millennial_2015  GenZ_2023  change
Obese                      19.7       23.4     3.8
No_exercise                17.5       16.4    -1.1
Smoker                     17.4        6.8   -10.6
Binge_drinking             26.9       23.9    -3.1
diabetes                    1.5        2.8     1.2

STEP 5  Gen Z odds of diabetes compared with Millennials, training set (1 = no difference):
                             GenZ_odds_ratio  low_95  high_95
model                                                        
same age and sex                        1.83    1.62     2.07
+ same obesity                          1.71    1.51     1.93
+ same exercise                         1.72    1.52     1.95
+ same smoking and drinking             1.85    1.63     2.10

        validation check (lower log loss = better fit to new people):
                      model      split  log_loss    AUC
           same age and sex validation    0.0923 0.5523
             + same obesity validation    0.0894 0.6450
            + same exercise validation    0.0895 0.6472
+ same smoking and drinking validation    0.0896 0.6449
        best-fitting model: + same obesity

STEP 6  test set: AUC = 0.692, log loss = 0.0969
```
