# Q1 part 2: exercise vs weight, smoking, drinking and diet
# BRFSS 2015, all factors in one model

import pandas as pd
import tools

factors = ["Obese", "Overweight", "No_exercise", "Smoker", "Heavy_drinking", "Little_fruit", "Little_vegetables"]


def add_factors(d):
    # 1 = has the risk
    d["Obese"] = (d.BMI >= 30).astype(int)
    d["Overweight"] = ((d.BMI >= 25) & (d.BMI < 30)).astype(int)
    d["No_exercise"] = 1 - d.PhysActivity
    d["Heavy_drinking"] = d.HvyAlcoholConsump
    d["Little_fruit"] = 1 - d.Fruits
    d["Little_vegetables"] = 1 - d.Veggies
    return d


train, valid = tools.load_train_valid("q1")
train, valid = add_factors(train), add_factors(valid)

print("\n% with diabetes (train):")
for f in factors:
    yes = tools.wpct(train.Diabetes[train[f] == 1], train.w[train[f] == 1])
    no = tools.wpct(train.Diabetes[train[f] == 0], train.w[train[f] == 0])
    print(f"{f:18s} yes {yes:5.1f}   no {no:5.1f}")

fx = " + ".join(factors)
models = {"factors only": f"Diabetes ~ {fx}",
          "factors + demographics": f"Diabetes ~ {fx} + C(Age) + Sex + C(Income) + C(Education)"}
fits, scores, best = tools.fit_and_pick(models, train, valid, "Diabetes")
print("\nvalidation:")
print(pd.DataFrame(scores).round(4).to_string(index=False))
print("picked:", best)

m = fits[best]
test = add_factors(tools.load_test("q1"))
scores.append(tools.score(m, test, "Diabetes", best, "test"))
print(f"\ntest auc {scores[-1]['auc']:.3f}, log loss {scores[-1]['log_loss']:.4f}")

out = tools.odds_ratios(m, factors).sort_values("odds_ratio", ascending=False)
print("\nodds ratios:")
print(out.round(2).to_string())

out.to_csv("results/q1_part2_exercise_vs_factors.csv")
pd.DataFrame(scores).to_csv("results/q1_part2_model_check.csv", index=False)
tools.barh(out.odds_ratio, "figures/q1_part2_exercise_vs_factors.png", "Odds ratio", highlight="No_exercise")
