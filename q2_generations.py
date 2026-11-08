# Q2: Gen Z (2023) vs Millennials (2015), ages 18-29
# does adding habits to the model shrink the Gen Z odds ratio?

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tools

d = tools.load_young()
print(d.groupby("year").size().to_dict())

d["GenZ"] = (d.year == 2023).astype(int)
d["Obese"] = (d.weight_status == "Obese").astype(int)
d["No_exercise"] = d.inactive
d["Smoker"] = d.current_smoker
d["Binge_drinking"] = d.binge
habits = ["Obese", "No_exercise", "Smoker", "Binge_drinking"]

# stratify on year and diabetes together
grp = d.year.astype(str) + "_" + d.diabetes.astype(int).astype(str)
train, valid, test = tools.split(d, grp)
print("train", len(train), "valid", len(valid), "test", len(test))

old, new = train[train.GenZ == 0], train[train.GenZ == 1]
pct = pd.DataFrame({c: {"2015": tools.wpct(old[c], old.w), "2023": tools.wpct(new[c], new.w)}
                    for c in habits + ["diabetes"]}).T
pct["change"] = pct["2023"] - pct["2015"]
print("\n% of young adults (train):")
print(pct.round(1).to_string())

steps = {"age, sex": [],
         "+ obesity": ["Obese"],
         "+ exercise": ["Obese", "No_exercise"],
         "+ smoking, drinking": ["Obese", "No_exercise", "Smoker", "Binge_drinking"]}
models = {name: "diabetes ~ GenZ + male + age_25_29" + "".join(" + " + h for h in extra)
          for name, extra in steps.items()}
fits, scores, best = tools.fit_and_pick(models, train, valid, "diabetes")

out = pd.concat([tools.odds_ratios(m, ["GenZ"]).rename(index={"GenZ": name}) for name, m in fits.items()])
out.index.name = "model"
out.columns = ["genz_odds_ratio", "low_95", "high_95"]
print("\nGen Z odds ratio (train):")
print(out.round(2).to_string())

print("\nvalidation:")
print(pd.DataFrame(scores).round(4).to_string(index=False))
print("picked:", best)

scores.append(tools.score(fits[best], test, "diabetes", best, "test"))
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
