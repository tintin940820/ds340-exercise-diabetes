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
