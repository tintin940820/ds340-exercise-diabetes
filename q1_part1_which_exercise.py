# Q1 part 1: which exercise is linked to the lowest odds of diabetes?
# BRFSS 2015, each person's main exercise vs no exercise

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = pd.read_csv("data/analysis_core.csv.gz")
d["w"] = d.weight / d.weight.mean()

# main sport codes from the BRFSS codebook, 0 = no exercise
names = {2: "Aerobics", 5: "Basketball", 6: "Exercise bike", 7: "Bicycling", 11: "Calisthenics", 14: "Dancing",
         15: "Elliptical", 18: "Gardening", 19: "Golf (cart)", 20: "Golf (walking)", 22: "Hiking", 28: "Jogging",
         31: "Mowing lawn", 37: "Running", 54: "Stair climbing", 57: "Swimming", 58: "Swimming laps",
         61: "Tennis", 64: "Walking", 67: "Weight lifting", 69: "Yoga", 73: "Household chores", 76: "Yard work"}
d["sport"] = d.MainSport.map(names)
d.loc[d.MainSport == 0, "sport"] = "No exercise"
d = d.dropna(subset=["sport"])
counts = d.sport.value_counts()
d = d[d.sport.map(counts) >= 500]
print(len(d), "people,", d.sport.nunique() - 1, "sports")

# 70% train, 10% validation, 20% test
train, rest = train_test_split(d, test_size=0.30, stratify=d.Diabetes, random_state=42)
valid, test = train_test_split(rest, test_size=2 / 3, stratify=rest.Diabetes, random_state=42)
print("train", len(train), "valid", len(valid), "test", len(test))

pct = train.groupby("sport").apply(lambda g: 100 * np.average(g.Diabetes, weights=g.w))
print("\n% with diabetes (train):")
print(pct.sort_values().round(1).to_string())

sport = "C(sport, Treatment('No exercise'))"
models = {"basic": f"Diabetes ~ {sport} + C(Age) + Sex",
          "full": f"Diabetes ~ {sport} + C(Age) + Sex + C(Income) + C(Education) + Smoker + HvyAlcoholConsump + Fruits + Veggies"}
fits, scores = {}, []
for name, f in models.items():
    fits[name] = smf.glm(f, data=train, family=sm.families.Binomial(), var_weights=train.w).fit()
    p = fits[name].predict(valid)
    scores.append({"model": name, "set": "valid", "log_loss": log_loss(valid.Diabetes, p, sample_weight=valid.w),
                   "auc": roc_auc_score(valid.Diabetes, p, sample_weight=valid.w)})
best = min(scores, key=lambda s: s["log_loss"])["model"]
print("\nvalidation:")
print(pd.DataFrame(scores).round(4).to_string(index=False))
print("picked:", best)

m = fits[best]
p = m.predict(test)
scores.append({"model": best, "set": "test", "log_loss": log_loss(test.Diabetes, p, sample_weight=test.w),
               "auc": roc_auc_score(test.Diabetes, p, sample_weight=test.w)})
print(f"\ntest auc {scores[-1]['auc']:.3f}, log loss {scores[-1]['log_loss']:.4f}")

rows = [k for k in m.params.index if k.startswith("C(sport")]
ci = np.exp(m.conf_int().loc[rows])
out = pd.DataFrame({"odds_ratio": np.exp(m.params[rows]), "low_95": ci[0], "high_95": ci[1]})
out.index = [k.split("T.")[1].rstrip("]") for k in rows]
out["n_train"] = train.sport.value_counts()[out.index]
out = out.sort_values("odds_ratio")
print("\nodds ratio vs no exercise:")
print(out.round(2).to_string())

out.to_csv("results/q1_part1_which_exercise.csv")
pd.DataFrame(scores).to_csv("results/q1_part1_model_check.csv", index=False)
fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.barh(out.index[::-1], out.odds_ratio[::-1], color="#1f63a8")
ax.axvline(1, color="#c4622d")
ax.set_xlabel("Odds ratio vs no exercise")
fig.savefig("figures/q1_part1_which_exercise.png", dpi=200, bbox_inches="tight")
