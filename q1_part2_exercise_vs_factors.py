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
