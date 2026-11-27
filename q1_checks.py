# two checks on Q1 part 1, from the midterm peer review:
#   1. reverse cause: people who are already sick may exercise less (or switch to walking).
#      refit the same model without people in fair/poor health or with trouble walking.
#   2. amount, not only type: the survey also gives minutes of activity per week.
#      fit the same model with minutes per week in groups instead of the main sport.
# both use the training set only (like the odds ratios in q1_part1), so the test set is not used here.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tools

train, _ = tools.load_train_valid("q1")

names = {0: "No exercise", 7: "Bicycling", 37: "Running", 61: "Tennis", 64: "Walking",
         67: "Weight lifting", 69: "Yoga", 18: "Gardening", 2: "Aerobics", 20: "Golf (walking)", 5: "Basketball"}
adjust = "C(Age) + Sex + C(Income) + C(Education) + Smoker + HvyAlcoholConsump + Fruits + Veggies"

# ---- check 1: same model as q1_part1 (full), without people in fair/poor health or with trouble walking
d = train[train.MainSport.isin(names)].copy()
d["sport"] = d.MainSport.map(names)
healthy = d[(d.GenHlth <= 3) & (d.DiffWalk == 0)]
print(f"\ncheck 1: {len(d)} people -> {len(healthy)} after removing fair/poor health or trouble walking")
rows = {}
for label, data in [("all", d), ("healthier only", healthy)]:
    m = tools.fit_logit(f"Diabetes ~ C(sport, Treatment('No exercise')) + {adjust}", data)
    keys = [k for k in m.params.index if k.startswith("C(sport")]
    out = tools.odds_ratios(m, keys)
    out.index = [k.split("T.")[1].rstrip("]") for k in keys]
    rows[label] = out
check1 = pd.DataFrame({"all": rows["all"].odds_ratio, "healthier only": rows["healthier only"].odds_ratio,
                       "low_95": rows["healthier only"].low_95, "high_95": rows["healthier only"].high_95})
check1 = check1.sort_values("all")
check1["change"] = check1["healthier only"] - check1["all"]
print("odds ratio vs no exercise:")
print(check1.round(2).to_string())

# ---- check 2: minutes of activity per week instead of the main sport
bins = [-1, 0, 149, 299, 100000]
labels = ["0 min", "1-149 min", "150-299 min", "300+ min"]
m2 = train.dropna(subset=["PA_minutes"]).copy()
m2["minutes"] = pd.cut(m2.PA_minutes, bins=bins, labels=labels).astype(str)
fit2 = tools.fit_logit(f"Diabetes ~ C(minutes, Treatment('0 min')) + {adjust}", m2)
keys = [k for k in fit2.params.index if k.startswith("C(minutes")]
check2 = tools.odds_ratios(fit2, keys)
check2.index = [k.split("T.")[1].rstrip("]") for k in keys]
check2["n_train"] = m2.minutes.value_counts()[check2.index].values
print("\ncheck 2: odds ratio vs 0 minutes per week:")
print(check2.round(2).to_string())

check1.to_csv("results/q1_check_healthier_only.csv")
check2.to_csv("results/q1_check_minutes_per_week.csv")

# figure: each sport's odds ratio with everyone vs without people in fair/poor health or with trouble walking
fig, ax = plt.subplots(figsize=(6.5, 4))
y = np.arange(len(check1))
ax.barh(y - 0.2, check1["all"], height=0.4, color="#9a9a9a", label="everyone")
ax.barh(y + 0.2, check1["healthier only"], height=0.4, color="#1f63a8", label="without fair/poor health or trouble walking")
ax.set_yticks(y)
ax.set_yticklabels(check1.index)
ax.invert_yaxis()
ax.axvline(1, color="#222222")
ax.set_xlabel("Odds ratio vs no exercise")
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8)
fig.savefig("figures/q1_check_healthier_only.png", dpi=200, bbox_inches="tight")
