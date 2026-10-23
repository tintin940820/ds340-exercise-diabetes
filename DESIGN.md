# DS340: Code Design

## The questions
- **Q1 part 1.** Which exercise helps the most against diabetes?
- **Q1 part 2.** How does exercise compare with other factors (weight, smoking, drinking, diet)?
- **Q2.** How do Gen Z and Millennial young adults compare (habits, diabetes, and whether habits explain the difference)?

## Folder layout
```
prepare/                          reads the raw CDC files once and writes the two clean tables (unchanged, already run)
data/analysis_core.csv.gz         2015, all adults (253,680)       -> used by Q1
data/generation_core.csv.gz       2015/2021/2023, ages 18 to 29    -> used by Q2
q1_part1_which_exercise.py        Q1 part 1
q1_part2_exercise_vs_factors.py   Q1 part 2
q2_generations.py                 Q2
run_all.sh                        runs the three scripts
results/                          tables (.csv) and printouts (log_*.txt)
figures/                          one chart per question
_old/                             the previous version, moved here and not deleted
```

## The design in one sentence
**There is one script per question, and every script follows the same five steps:**

| Step | What it does | Why |
|---|---|---|
| 1. Load | Reads the clean table and adds the CDC survey weight | The weight makes the answer describe all US adults, not just the people who happened to answer |
| 2. Make 0/1 columns | Turns each thing we compare into a yes/no column (or one label, such as "main sport") | One coding rule makes every odds ratio mean the same thing |
| 3. Simple look | Shows the percent with diabetes in each group, with no model | A plain number anyone can check, and a baseline to compare the model against |
| 4. Fair comparison | Runs one weighted logistic regression and reports odds ratios with 95% intervals | Exercisers are younger, richer and smoke less. The model holds those things equal so we compare like with like |
| 5. Save | Writes a .csv table and one .png chart | The paper uses these files directly, so no number is typed by hand |

## Why this design fits our questions

**Why one script per question.** Each question uses different data and asks something different. Keeping them apart means:
- you can run or read one question without the others
- a change to one question cannot break another
- when you present, each file is one slide's worth of work

**Why the same five steps everywhere.** Once you understand one script you understand all three. The only part that changes between questions is **step 2 (what we compare)** and **step 4 (what goes into the model)**.

**Why logistic regression.** The outcome is yes or no (diabetes or prediabetes). Logistic regression is the standard model for a yes/no outcome. It gives an **odds ratio**, which reads simply:
- 1 means no difference
- below 1 means less diabetes
- above 1 means more diabetes

It is also what our parent papers used, so our results can be compared with theirs.

**Why survey weights.** BRFSS deliberately surveys some groups more than others. Without the weight the percentages would be off. We divide the weight by its average so it only changes how much each person counts, not the size of the sample.

### Q1 part 1: which exercise helps most
- **Step 2:** each person gets one label, either their main sport or "No exercise". We keep only sports with at least 500 people, because a sport with 40 people would give a noisy result that could top the list by luck.
- **Step 4:** a single model with "sport" as one variable and **"No exercise" as the reference group**. Every sport is then compared with the same baseline, so the odds ratios can be ranked directly.
- **What we hold equal:** age, sex, income, education, smoking, drinking and diet. These are the things that differ between, say, runners and walkers and that also affect diabetes. Without them, running would look good simply because runners are young.
- **What we leave out on purpose:** body weight (BMI). Exercise partly works *through* weight. Holding weight equal would hide part of the benefit we are trying to measure.

### Q1 part 2: exercise vs other factors
- **Step 2:** every factor is coded as **1 = has the risk** (obese, no exercise, smoker, little fruit, and so on). That is why exercise appears as "No_exercise". Coded this way, every odds ratio points the same direction and you can rank them by size.
- **Step 4:** all the factors go into **one model together**. If each factor got its own model, obesity would get credit for some of the effect of not exercising, and the other way round. In one model, each factor's number is its own effect with the others held equal.
- Here BMI *is* in the model, because weight is one of the factors we are comparing.

### Q2: generations
- **Why compare 2015 with 2023, ages 18 to 29.** Young adults in 2015 were Millennials and young adults in 2023 are Gen Z. Because both groups are the same age, any difference comes from the generation, not from getting older.
- **Step 1:** we keep only people who answered every question used here, so every model in step 4 uses exactly the same people. Otherwise a change in the odds ratio could come from a change in who is counted.
- **Step 4 is the key design choice.** We fit the Gen Z odds ratio once with only age and sex, then **add habits one step at a time**:
  - if the odds ratio moves toward 1 when a habit is added, that habit explains part of the gap
  - if it doesn't move, it doesn't explain the gap

  This is the simplest way to ask "is it because of their habits?" with a single number to watch.

## What the results say
- **Q1 part 1.** Every exercise beats no exercise. Running, walking golf, tennis, basketball, weight lifting and yoga are linked to about two thirds lower odds (odds ratio about 0.3). Walking is the most common exercise but among the weakest (0.80). Exercise bike and stair climbing barely differ from no exercise.
- **Q1 part 2.** Weight matters most: obese 4.12 and overweight 1.86. Not exercising comes next at 1.28, which is larger than smoking (1.14) or diet (about 1.07 to 1.09). Heavy drinking shows 0.56. This is a known quirk of survey data (heavy drinkers who are sick tend to stop drinking), so it should not be read as drinking being protective.
- **Q2.** Gen Z young adults have more diabetes or prediabetes: 2.6% vs 1.6%, odds ratio 1.71.
  - **Habits changed in both directions:** obesity is up 4 points, while smoking is down 10.5 points and binge drinking is down 3.8 points.
  - **What explains the gap:** holding obesity equal shrinks the odds ratio from 1.71 to 1.59, so rising obesity explains part of it. Adding smoking and drinking pushes it back up to 1.73, because Gen Z smokes far less, which should have *lowered* their risk.
  - **Bottom line:** habits as a whole do not explain why Gen Z has more diabetes. Obesity explains some of it, and the rest is something these questions do not measure (for example more testing, other diet changes, or sleep).

## Limits to mention in the paper
- **The data is a survey taken at one point in time.** These are associations, not proof that a sport causes lower diabetes. For example, people who are already healthy may be more likely to choose running.
- **The 95% intervals are a little too narrow.** They use the weights but not the survey's cluster design. The conclusions don't change, but the intervals would be somewhat wider with the full survey method used in the earlier version.
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
Data: CDC BRFSS 2015, 253,680 US adults. Compares each person's main exercise with doing no exercise."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
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

# STEP 3. Simple look: percent with diabetes in each group
raw = d.groupby("sport").apply(lambda g: 100 * np.average(g.Diabetes, weights=g.w))
print("\nSTEP 3  percent with diabetes or prediabetes, before any adjustment:")
print(raw.sort_values().round(1).to_string())

# STEP 4. Fair comparison: logistic regression, holding age, sex, income, education, smoking, drinking, diet equal
model = smf.glm("Diabetes ~ C(sport, Treatment('No exercise')) + C(Age) + Sex + C(Income) + C(Education)"
                " + Smoker + HvyAlcoholConsump + Fruits + Veggies",
                data=d, family=sm.families.Binomial(), var_weights=d.w).fit()
rows = [k for k in model.params.index if k.startswith("C(sport")]
table = pd.DataFrame({"odds_ratio": np.exp(model.params[rows]),
                      "low_95": np.exp(model.conf_int().loc[rows, 0]),
                      "high_95": np.exp(model.conf_int().loc[rows, 1])})
table.index = [k.split("T.")[1].rstrip("]") for k in rows]
table["people"] = counts[table.index]
table = table.sort_values("odds_ratio")
print("\nSTEP 4  odds ratio vs no exercise (1 = same as no exercise, lower = more protective):")
print(table.round(2).to_string())

# STEP 5. Save the table and a chart
table.to_csv("results/q1_part1_which_exercise.csv")
fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.barh(table.index[::-1], table.odds_ratio[::-1], color="#1f63a8")
ax.axvline(1, color="#c4622d")
ax.set_xlabel("Odds of diabetes compared with no exercise (lower = more protective)")
fig.savefig("figures/q1_part1_which_exercise.png", dpi=200, bbox_inches="tight")
```

### q1_part2_exercise_vs_factors.py
```python
"""Question 1, part 2: How does exercise compare with other factors (weight, smoking, drinking, diet)?
Data: CDC BRFSS 2015, 253,680 US adults. All factors go into one model so they can be ranked side by side."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
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

# STEP 3. Simple look: percent with diabetes with and without each factor
print("STEP 3  percent with diabetes or prediabetes, before any adjustment:")
for f in factors:
    has = 100 * np.average(d.Diabetes[d[f] == 1], weights=d.w[d[f] == 1])
    no = 100 * np.average(d.Diabetes[d[f] == 0], weights=d.w[d[f] == 0])
    print(f"        {f:18s} has it: {has:5.1f}%   does not: {no:5.1f}%")

# STEP 4. One model with every factor together, plus age, sex, income, education
model = smf.glm("Diabetes ~ " + " + ".join(factors) + " + C(Age) + Sex + C(Income) + C(Education)",
                data=d, family=sm.families.Binomial(), var_weights=d.w).fit()
table = pd.DataFrame({"odds_ratio": np.exp(model.params[factors]),
                      "low_95": np.exp(model.conf_int().loc[factors, 0]),
                      "high_95": np.exp(model.conf_int().loc[factors, 1])}).sort_values("odds_ratio", ascending=False)
print("\nSTEP 4  odds ratio for each factor, all in one model (above 1 = more diabetes, bigger = stronger):")
print(table.round(2).to_string())

# STEP 5. Save the table and a chart (exercise highlighted)
table.to_csv("results/q1_part2_exercise_vs_factors.csv")
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
How do their habits differ, who has more diabetes, and do the habits explain the difference?"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
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

# STEP 3. Simple look: percent with each habit, and with diabetes, in each generation
rows = {}
for col in habits + ["diabetes"]:
    rows[col] = {gen: 100 * np.average(g[col], weights=g.w) for gen, g in
                 (("Millennial_2015", d[d.GenZ == 0]), ("GenZ_2023", d[d.GenZ == 1]))}
pct = pd.DataFrame(rows).T
pct["change"] = pct.GenZ_2023 - pct.Millennial_2015
print("\nSTEP 3  percent of young adults:")
print(pct.round(1).to_string())

# STEP 4. Gen Z odds ratio: first same age and sex, then add habits one step at a time.
# If the odds ratio moves toward 1 when a habit is added, that habit explains part of the gap.
steps = {"same age and sex": [],
         "+ same obesity": ["Obese"],
         "+ same exercise": ["Obese", "No_exercise"],
         "+ same smoking and drinking": ["Obese", "No_exercise", "Smoker", "Binge_drinking"]}
out = []
for name, extra in steps.items():
    model = smf.glm("diabetes ~ GenZ + male + age_25_29" + "".join(" + " + h for h in extra),
                    data=d, family=sm.families.Binomial(), var_weights=d.w).fit()
    ci = np.exp(model.conf_int().loc["GenZ"])
    out.append({"model": name, "GenZ_odds_ratio": np.exp(model.params["GenZ"]), "low_95": ci[0], "high_95": ci[1]})
table = pd.DataFrame(out).set_index("model")
print("\nSTEP 4  Gen Z odds of diabetes compared with Millennials (1 = no difference):")
print(table.round(2).to_string())

# STEP 5. Save the tables and a chart
pct.to_csv("results/q2_habits_and_diabetes.csv")
table.to_csv("results/q2_genz_odds_ratio.csv")
fig, ax = plt.subplots(figsize=(6.5, 3.5))
y = np.arange(len(pct))
ax.barh(y - 0.2, pct.Millennial_2015, height=0.4, color="#9a9a9a", label="Millennials (2015)")
ax.barh(y + 0.2, pct.GenZ_2023, height=0.4, color="#1f63a8", label="Gen Z (2023)")
ax.set_yticks(y)
ax.set_yticklabels(pct.index)
ax.invert_yaxis()
ax.set_xlabel("Percent of young adults aged 18 to 29")
ax.legend(frameon=False)
fig.savefig("figures/q2_generations.png", dpi=200, bbox_inches="tight")
```

## What the code printed

### results/log_q1_part1_which_exercise.txt
```
STEP 2  238404 people, 23 sports (plus 'No exercise')

STEP 3  percent with diabetes or prediabetes, before any adjustment:
sport
Running              2.6
Basketball           3.5
Weight lifting       4.1
Yoga                 4.8
Jogging              5.4
Tennis               5.7
Golf (walking)       5.9
Elliptical           7.1
Hiking               7.2
Bicycling            8.1
Swimming laps        8.6
Aerobics             8.8
Calisthenics        10.6
Dancing             10.6
Gardening           13.1
Swimming            13.1
Golf (cart)         14.3
Walking             15.4
Yard work           17.0
Mowing lawn         17.0
Exercise bike       17.4
Household chores    18.3
Stair climbing      18.6
No exercise         21.1

STEP 4  odds ratio vs no exercise (1 = same as no exercise, lower = more protective):
                  odds_ratio  low_95  high_95  people
Golf (walking)          0.28    0.22     0.37    1227
Running                 0.31    0.28     0.34   11740
Tennis                  0.33    0.24     0.45     810
Basketball              0.35    0.26     0.46     892
Weight lifting          0.35    0.32     0.39    6794
Yoga                    0.35    0.29     0.44    1962
Bicycling               0.50    0.45     0.55    4989
Elliptical              0.50    0.44     0.57    3575
Jogging                 0.51    0.43     0.60    1718
Swimming laps           0.54    0.43     0.68    1076
Gardening               0.55    0.51     0.59   10958
Hiking                  0.59    0.50     0.71    1776
Aerobics                0.60    0.54     0.68    4756
Mowing lawn             0.62    0.50     0.76     790
Calisthenics            0.63    0.57     0.71    3050
Golf (cart)             0.66    0.58     0.74    2713
Dancing                 0.69    0.58     0.82    1204
Yard work               0.72    0.65     0.81    3153
Household chores        0.77    0.64     0.92    1095
Swimming                0.77    0.67     0.89    1784
Walking                 0.80    0.78     0.82  105401
Stair climbing          0.90    0.71     1.12     586
Exercise bike           0.91    0.83     0.99    4595
```

### results/log_q1_part2_exercise_vs_factors.txt
```
STEP 3  percent with diabetes or prediabetes, before any adjustment:
        Obese              has it:  23.7%   does not:   9.6%
        Overweight         has it:  12.5%   does not:  15.0%
        No_exercise        has it:  21.1%   does not:  11.8%
        Smoker             has it:  17.1%   does not:  11.8%
        Heavy_drinking     has it:   7.2%   does not:  14.5%
        Little_fruit       has it:  15.7%   does not:  13.1%
        Little_vegetables  has it:  17.7%   does not:  13.2%

STEP 4  odds ratio for each factor, all in one model (above 1 = more diabetes, bigger = stronger):
                   odds_ratio  low_95  high_95
Obese                    4.12    3.98     4.27
Overweight               1.86    1.79     1.93
No_exercise              1.28    1.24     1.31
Smoker                   1.14    1.11     1.17
Little_vegetables        1.09    1.06     1.13
Little_fruit             1.07    1.04     1.09
Heavy_drinking           0.56    0.52     0.60
```

### results/log_q2_generations.txt
```
STEP 1  young adults: {2015: 35367, 2023: 40371}

STEP 3  percent of young adults:
                Millennial_2015  GenZ_2023  change
Obese                      19.6       23.6     4.0
No_exercise                17.7       16.3    -1.3
Smoker                     17.3        6.8   -10.5
Binge_drinking             27.3       23.5    -3.8
diabetes                    1.6        2.6     1.1

STEP 4  Gen Z odds of diabetes compared with Millennials (1 = no difference):
                             GenZ_odds_ratio  low_95  high_95
model                                                        
same age and sex                        1.71    1.55     1.90
+ same obesity                          1.59    1.43     1.76
+ same exercise                         1.60    1.44     1.77
+ same smoking and drinking             1.73    1.55     1.92
```
