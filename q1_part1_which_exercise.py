# Q1 part 1: which exercise is linked to the lowest odds of diabetes?
# BRFSS 2015, each person's main exercise vs no exercise

import pandas as pd
import tools

d = tools.load_adults()

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

train, valid, test = tools.split(d, d.Diabetes)
print("train", len(train), "valid", len(valid), "test", len(test))

pct = train.groupby("sport").apply(lambda g: tools.wpct(g.Diabetes, g.w))
print("\n% with diabetes (train):")
print(pct.sort_values().round(1).to_string())

sport = "C(sport, Treatment('No exercise'))"
models = {"basic": f"Diabetes ~ {sport} + C(Age) + Sex",
          "full": f"Diabetes ~ {sport} + C(Age) + Sex + C(Income) + C(Education) + Smoker + HvyAlcoholConsump + Fruits + Veggies"}
fits, scores, best = tools.fit_and_pick(models, train, valid, "Diabetes")
print("\nvalidation:")
print(pd.DataFrame(scores).round(4).to_string(index=False))
print("picked:", best)

m = fits[best]
scores.append(tools.score(m, test, "Diabetes", best, "test"))
print(f"\ntest auc {scores[-1]['auc']:.3f}, log loss {scores[-1]['log_loss']:.4f}")

rows = [k for k in m.params.index if k.startswith("C(sport")]
out = tools.odds_ratios(m, rows)
out.index = [k.split("T.")[1].rstrip("]") for k in rows]
out["n_train"] = train.sport.value_counts()[out.index]
out = out.sort_values("odds_ratio")
print("\nodds ratio vs no exercise:")
print(out.round(2).to_string())

out.to_csv("results/q1_part1_which_exercise.csv")
pd.DataFrame(scores).to_csv("results/q1_part1_model_check.csv", index=False)
tools.barh(out.odds_ratio, "figures/q1_part1_which_exercise.png", "Odds ratio vs no exercise", size=(6.5, 6.5))
