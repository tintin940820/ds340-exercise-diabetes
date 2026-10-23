# recode the young adult file into one clean table
# 2015 = Millennials, 2023 = Gen Z (2023 has no fruit/veg questions)
import numpy as np
import pandas as pd

r = pd.read_csv("../data/generation_raw_young.csv.gz")
d = pd.DataFrame({"year": r.year})
two = lambda s, yes, no: s.map({yes: 1, no: 0})

d["diabetes"] = r.diab_raw.map({1: 1, 4: 1, 2: 0, 3: 0})
d["diabetes_dx"] = r.diab_raw.map({1: 1, 3: 0, 2: 0})
d["inactive"] = two(r.exercise, 2, 1)
d["bmi"] = r.bmi / 100
d["weight_status"] = pd.cut(d.bmi, [0, 24.999, 29.999, 200], labels=["Normal or under", "Overweight", "Obese"])
d["current_smoker"] = r.smoker4.map({1: 1, 2: 1, 3: 0, 4: 0})
d["binge"] = two(r.binge, 2, 1)
d["heavy_drink"] = two(r.heavy_drink, 2, 1)
d["fruit_lt_daily"] = two(r.fruit_lt1, 2, 1)
d["veg_lt_daily"] = two(r.veg_lt1, 2, 1)
d["diffwalk"] = two(r.diffwalk, 1, 2)
d["male"] = two(r.sex, 1, 2)
d["age_25_29"] = r.age5.map({1: 0, 2: 1})
# income codes changed after 2015, so use 3 groups
inc = r.income
d["income_grp"] = np.select([inc.between(1, 4), inc.between(5, 6), inc.between(7, 11)],
                            ["Under 25k", "25k to 50k", "50k and over"], default="Unknown")
d["college"] = r.educ.map({1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1})
d["weight"], d["strata"], d["psu"] = r.weight, r.strata, r.psu
d.to_csv("../data/generation_core.csv.gz", index=False)
t = d.groupby("year").agg(n=("diabetes", "size"), outcome_known=("diabetes", "count"))
print(t)
print(d.groupby("year")[["diabetes", "inactive", "current_smoker", "binge", "fruit_lt_daily", "veg_lt_daily"]].mean().round(3))
