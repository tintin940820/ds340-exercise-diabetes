"""Question 1, part 2: How does exercise compare with other factors (weight, smoking, drinking, diet)?
Data: CDC BRFSS 2015, 253,680 US adults. All factors go into one model so they can be ranked side by side."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
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

# STEP 3. Simple look: percent with diabetes with and without each factor
print("STEP 3  percent with diabetes or prediabetes, before any adjustment:")
for f in factors:
    has = 100 * np.average(d.Diabetes[d[f] == 1], weights=d.w[d[f] == 1])
    no = 100 * np.average(d.Diabetes[d[f] == 0], weights=d.w[d[f] == 0])
    print(f"        {f:18s} has it: {has:5.1f}%   does not: {no:5.1f}%")

# STEP 4. One model with every factor together, plus age, sex, income, education
model = smf.glm("Diabetes ~ " + " + ".join(factors) + " + C(Age) + Sex + C(Income) + C(Education)",
                data=d, family=sm.families.Binomial(), var_weights=d.w).fit()
table = pd.DataFrame({"odds_ratio": np.exp(model.params[factors]),
                      "low_95": np.exp(model.conf_int().loc[factors, 0]),
                      "high_95": np.exp(model.conf_int().loc[factors, 1])}).sort_values("odds_ratio", ascending=False)
print("\nSTEP 4  odds ratio for each factor, all in one model (above 1 = more diabetes, bigger = stronger):")
print(table.round(2).to_string())

# STEP 5. Save the table and a chart (exercise highlighted)
table.to_csv("results/q1_part2_exercise_vs_factors.csv")
fig, ax = plt.subplots(figsize=(6.5, 3.5))
colors = ["#c4622d" if f == "No_exercise" else "#1f63a8" for f in table.index[::-1]]
ax.barh(table.index[::-1], table.odds_ratio[::-1], color=colors)
ax.axvline(1, color="#222222")
ax.set_xlabel("Odds ratio for diabetes (above 1 = more diabetes)")
fig.savefig("figures/q1_part2_exercise_vs_factors.png", dpi=200, bbox_inches="tight")
