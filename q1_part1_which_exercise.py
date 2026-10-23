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
