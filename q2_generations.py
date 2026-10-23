"""Question 2: Gen Z vs Millennial young adults (ages 18 to 29).
Millennials = young adults surveyed in 2015. Gen Z = young adults surveyed in 2023. Same age, different generation.
How do their habits differ, who has more diabetes, and do the habits explain the difference?"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
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

# STEP 3. Simple look: percent with each habit, and with diabetes, in each generation
rows = {}
for col in habits + ["diabetes"]:
    rows[col] = {gen: 100 * np.average(g[col], weights=g.w) for gen, g in
                 (("Millennial_2015", d[d.GenZ == 0]), ("GenZ_2023", d[d.GenZ == 1]))}
pct = pd.DataFrame(rows).T
pct["change"] = pct.GenZ_2023 - pct.Millennial_2015
print("\nSTEP 3  percent of young adults:")
print(pct.round(1).to_string())

# STEP 4. Gen Z odds ratio: first same age and sex, then add habits one step at a time.
# If the odds ratio moves toward 1 when a habit is added, that habit explains part of the gap.
steps = {"same age and sex": [],
         "+ same obesity": ["Obese"],
         "+ same exercise": ["Obese", "No_exercise"],
         "+ same smoking and drinking": ["Obese", "No_exercise", "Smoker", "Binge_drinking"]}
out = []
for name, extra in steps.items():
    model = smf.glm("diabetes ~ GenZ + male + age_25_29" + "".join(" + " + h for h in extra),
                    data=d, family=sm.families.Binomial(), var_weights=d.w).fit()
    ci = np.exp(model.conf_int().loc["GenZ"])
    out.append({"model": name, "GenZ_odds_ratio": np.exp(model.params["GenZ"]), "low_95": ci[0], "high_95": ci[1]})
table = pd.DataFrame(out).set_index("model")
print("\nSTEP 4  Gen Z odds of diabetes compared with Millennials (1 = no difference):")
print(table.round(2).to_string())

# STEP 5. Save the tables and a chart
pct.to_csv("results/q2_habits_and_diabetes.csv")
table.to_csv("results/q2_genz_odds_ratio.csv")
fig, ax = plt.subplots(figsize=(6.5, 3.5))
y = np.arange(len(pct))
ax.barh(y - 0.2, pct.Millennial_2015, height=0.4, color="#9a9a9a", label="Millennials (2015)")
ax.barh(y + 0.2, pct.GenZ_2023, height=0.4, color="#1f63a8", label="Gen Z (2023)")
ax.set_yticks(y)
ax.set_yticklabels(pct.index)
ax.invert_yaxis()
ax.set_xlabel("Percent of young adults aged 18 to 29")
ax.legend(frameon=False)
fig.savefig("figures/q2_generations.png", dpi=200, bbox_inches="tight")
