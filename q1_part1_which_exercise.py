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
