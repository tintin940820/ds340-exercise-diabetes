# recode BRFSS 2015 answers into the clean table
# same 21 variables as the CDC Diabetes Health Indicators data, plus main sport
import numpy as np
import pandas as pd

raw = pd.read_csv("../data/brfss2015_selected.csv.gz")
d = pd.DataFrame(index=raw.index)


def recode(series, mapping):
    # codes not in the mapping (don't know, refused) become NaN
    return series.map(mapping)


# DIABETE3: 1 yes, 2 only during pregnancy, 3 no, 4 prediabetes
d["Diabetes"] = recode(raw.DIABETE3, {1: 1, 4: 1, 2: 0, 3: 0})
d["Diabetes_dx"] = recode(raw.DIABETE3, {1: 1, 3: 0})

d["HighBP"] = recode(raw._RFHYPE5, {1: 0, 2: 1})
d["HighChol"] = recode(raw.TOLDHI2, {1: 1, 2: 0})
d["CholCheck"] = recode(raw._CHOLCHK, {1: 1, 2: 0, 3: 0})
d["BMI"] = raw._BMI5 / 100.0
d["Smoker"] = recode(raw.SMOKE100, {1: 1, 2: 0})
d["Stroke"] = recode(raw.CVDSTRK3, {1: 1, 2: 0})
d["HeartDiseaseorAttack"] = recode(raw._MICHD, {1: 1, 2: 0})
d["PhysActivity"] = recode(raw._TOTINDA, {1: 1, 2: 0})
d["Fruits"] = recode(raw._FRTLT1, {1: 1, 2: 0})
d["Veggies"] = recode(raw._VEGLT1, {1: 1, 2: 0})
d["HvyAlcoholConsump"] = recode(raw._RFDRHV5, {1: 0, 2: 1})
d["AnyHealthcare"] = recode(raw.HLTHPLN1, {1: 1, 2: 0})
d["NoDocbcCost"] = recode(raw.MEDCOST, {1: 1, 2: 0})
d["GenHlth"] = raw.GENHLTH.where(raw.GENHLTH.between(1, 5))
d["MentHlth"] = raw.MENTHLTH.replace(88, 0).where(lambda s: s.between(0, 30))
d["PhysHlth"] = raw.PHYSHLTH.replace(88, 0).where(lambda s: s.between(0, 30))
d["DiffWalk"] = recode(raw.DIFFWALK, {1: 1, 2: 0})
d["Sex"] = recode(raw.SEX, {1: 1, 2: 0})  # 1 = male
d["Age"] = raw._AGEG5YR.where(raw._AGEG5YR.between(1, 13))
d["Education"] = raw.EDUCA.where(raw.EDUCA.between(1, 6))
d["Income"] = raw.INCOME2.where(raw.INCOME2.between(1, 8))

# exercise details
d["PA_level"] = raw._PACAT1.where(raw._PACAT1.between(1, 4))
d["PA_minutes"] = np.where(d.PhysActivity == 0, 0.0, raw.PA1MIN_)
d.loc[d.PhysActivity.isna(), "PA_minutes"] = np.nan
d["PA_minutes"] = d.PA_minutes.clip(upper=3000)
d["MainSport"] = np.where(d.PhysActivity == 0, 0, raw.EXRACT11)  # 0 = no exercise
d.loc[d.PhysActivity.isna() | raw.EXRACT11.isin([77, 99]), "MainSport"] = np.nan
d.loc[(d.PhysActivity == 1) & raw.EXRACT11.isna(), "MainSport"] = np.nan
d["MainSport_MET"] = raw.METVL11_ / 10.0

# STRENGTH: 101-199 per week, 201-299 per month, 888 never
s = raw.STRENGTH
per_week = np.where(s.between(101, 199), s - 100,
           np.where(s.between(201, 299), (s - 200) / 4.33,
           np.where(s == 888, 0, np.nan)))
d["Strength_per_week"] = np.clip(per_week, 0, 14)

# survey weight and design
d["weight"] = raw._LLCPWT
d["strata"] = raw._STSTR
d["psu"] = raw._PSU

STD21 = ["HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack",
         "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
         "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"]

core = d.dropna(subset=["Diabetes"] + STD21).copy()
print(len(d), "raw,", len(core), "complete")
core.to_csv("../data/analysis_core.csv.gz", index=False)
