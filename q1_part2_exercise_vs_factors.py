# Q1 part 2: exercise vs weight, smoking, drinking and diet
# BRFSS 2015, all factors in one model

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

# 1 = has the risk
d["Obese"] = (d.BMI >= 30).astype(int)
d["Overweight"] = ((d.BMI >= 25) & (d.BMI < 30)).astype(int)
d["No_exercise"] = 1 - d.PhysActivity
d["Heavy_drinking"] = d.HvyAlcoholConsump
d["Little_fruit"] = 1 - d.Fruits
d["Little_vegetables"] = 1 - d.Veggies
factors = ["Obese", "Overweight", "No_exercise", "Smoker", "Heavy_drinking", "Little_fruit", "Little_vegetables"]

# 70% train, 10% validation, 20% test
train, rest = train_test_split(d, test_size=0.30, stratify=d.Diabetes, random_state=42)
valid, test = train_test_split(rest, test_size=2 / 3, stratify=rest.Diabetes, random_state=42)
print("train", len(train), "valid", len(valid), "test", len(test))

print("\n% with diabetes (train):")
for f in factors:
    yes = 100 * np.average(train.Diabetes[train[f] == 1], weights=train.w[train[f] == 1])
    no = 100 * np.average(train.Diabetes[train[f] == 0], weights=train.w[train[f] == 0])
    print(f"{f:18s} yes {yes:5.1f}   no {no:5.1f}")

fx = " + ".join(factors)
models = {"factors only": f"Diabetes ~ {fx}",
          "factors + demographics": f"Diabetes ~ {fx} + C(Age) + Sex + C(Income) + C(Education)"}
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

ci = np.exp(m.conf_int().loc[factors])
out = pd.DataFrame({"odds_ratio": np.exp(m.params[factors]), "low_95": ci[0], "high_95": ci[1]})
out = out.sort_values("odds_ratio", ascending=False)
print("\nodds ratios:")
print(out.round(2).to_string())

out.to_csv("results/q1_part2_exercise_vs_factors.csv")
pd.DataFrame(scores).to_csv("results/q1_part2_model_check.csv", index=False)
fig, ax = plt.subplots(figsize=(6.5, 3.5))
colors = ["#c4622d" if f == "No_exercise" else "#1f63a8" for f in out.index[::-1]]
ax.barh(out.index[::-1], out.odds_ratio[::-1], color=colors)
ax.axvline(1, color="#222222")
ax.set_xlabel("Odds ratio")
fig.savefig("figures/q1_part2_exercise_vs_factors.png", dpi=200, bbox_inches="tight")
