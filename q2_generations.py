# Q2: Gen Z (2023) vs Millennials (2015), ages 18-29
# does adding habits to the model shrink the Gen Z odds ratio?

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = pd.read_csv("data/generation_core.csv.gz")
d = d[d.year.isin([2015, 2023])]
d = d.dropna(subset=["diabetes", "weight_status", "inactive", "current_smoker", "binge"])
d["w"] = d.weight / d.weight.mean()
print(d.groupby("year").size().to_dict())

d["GenZ"] = (d.year == 2023).astype(int)
d["Obese"] = (d.weight_status == "Obese").astype(int)
d["No_exercise"] = d.inactive
d["Smoker"] = d.current_smoker
d["Binge_drinking"] = d.binge
habits = ["Obese", "No_exercise", "Smoker", "Binge_drinking"]

# 70% train, 10% validation, 20% test, stratified on year and diabetes
grp = d.year.astype(str) + "_" + d.diabetes.astype(int).astype(str)
train, rest = train_test_split(d, test_size=0.30, stratify=grp, random_state=42)
valid, test = train_test_split(rest, test_size=2 / 3, stratify=grp[rest.index], random_state=42)
print("train", len(train), "valid", len(valid), "test", len(test))

pct = pd.DataFrame({c: {"2015": 100 * np.average(train[c][train.GenZ == 0], weights=train.w[train.GenZ == 0]),
                        "2023": 100 * np.average(train[c][train.GenZ == 1], weights=train.w[train.GenZ == 1])}
                    for c in habits + ["diabetes"]}).T
pct["change"] = pct["2023"] - pct["2015"]
print("\n% of young adults (train):")
print(pct.round(1).to_string())

steps = {"age, sex": [],
         "+ obesity": ["Obese"],
         "+ exercise": ["Obese", "No_exercise"],
         "+ smoking, drinking": ["Obese", "No_exercise", "Smoker", "Binge_drinking"]}
fits, rows, scores = {}, [], []
for name, extra in steps.items():
    m = smf.glm("diabetes ~ GenZ + male + age_25_29" + "".join(" + " + h for h in extra),
                data=train, family=sm.families.Binomial(), var_weights=train.w).fit()
    fits[name] = m
    ci = np.exp(m.conf_int().loc["GenZ"])
    rows.append({"model": name, "genz_odds_ratio": np.exp(m.params["GenZ"]), "low_95": ci[0], "high_95": ci[1]})
    p = m.predict(valid)
    scores.append({"model": name, "set": "valid", "log_loss": log_loss(valid.diabetes, p, sample_weight=valid.w),
                   "auc": roc_auc_score(valid.diabetes, p, sample_weight=valid.w)})
out = pd.DataFrame(rows).set_index("model")
print("\nGen Z odds ratio (train):")
print(out.round(2).to_string())

best = min(scores, key=lambda s: s["log_loss"])["model"]
print("\nvalidation:")
print(pd.DataFrame(scores).round(4).to_string(index=False))
print("picked:", best)

p = fits[best].predict(test)
scores.append({"model": best, "set": "test", "log_loss": log_loss(test.diabetes, p, sample_weight=test.w),
               "auc": roc_auc_score(test.diabetes, p, sample_weight=test.w)})
print(f"\ntest auc {scores[-1]['auc']:.3f}, log loss {scores[-1]['log_loss']:.4f}")

pct.to_csv("results/q2_habits_and_diabetes.csv")
out.to_csv("results/q2_genz_odds_ratio.csv")
pd.DataFrame(scores).to_csv("results/q2_model_check.csv", index=False)
fig, ax = plt.subplots(figsize=(6.5, 3.5))
y = np.arange(len(pct))
ax.barh(y - 0.2, pct["2015"], height=0.4, color="#9a9a9a", label="2015")
ax.barh(y + 0.2, pct["2023"], height=0.4, color="#1f63a8", label="2023")
ax.set_yticks(y)
ax.set_yticklabels(pct.index)
ax.invert_yaxis()
ax.set_xlabel("% of young adults (18-29)")
ax.legend(frameon=False)
fig.savefig("figures/q2_generations.png", dpi=200, bbox_inches="tight")
