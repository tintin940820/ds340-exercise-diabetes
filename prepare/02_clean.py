"""Step 2. Recode raw BRFSS 2015 answers into an analysis dataset.

The 21 predictors are rebuilt with the same rules used for the CDC Diabetes Health Indicators
dataset (the data used by Parent Paper 1), so results are directly comparable. On top of that we
keep the detailed exercise answers (type of activity, weekly minutes, activity level, muscle
strengthening) which the public 21-feature version throws away.
"""
import numpy as np
import pandas as pd

raw = pd.read_csv("../data/brfss2015_selected.csv.gz")
d = pd.DataFrame(index=raw.index)


def recode(series, mapping):
    """Map answer codes; anything not in the mapping (don't know / refused / missing) -> NaN."""
    return series.map(mapping)


# ---------- outcome ----------
# DIABETE3: 1 yes, 2 yes but only during pregnancy, 3 no, 4 pre-diabetes/borderline, 7/9 unknown
d["Diabetes"] = recode(raw.DIABETE3, {1: 1, 4: 1, 2: 0, 3: 0})          # same definition as Parent Paper 1
d["Diabetes_dx"] = recode(raw.DIABETE3, {1: 1, 3: 0})                   # diagnosed diabetes only (sensitivity)

# ---------- the 21 standard predictors ----------
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
d["Sex"] = recode(raw.SEX, {1: 1, 2: 0})                                   # 1 = male
d["Age"] = raw._AGEG5YR.where(raw._AGEG5YR.between(1, 13))                # 13 five-year groups
d["Education"] = raw.EDUCA.where(raw.EDUCA.between(1, 6))
d["Income"] = raw.INCOME2.where(raw.INCOME2.between(1, 8))

# ---------- detailed exercise variables (not in the public 21-feature dataset) ----------
d["PA_level"] = raw._PACAT1.where(raw._PACAT1.between(1, 4))             # 1 highly active ... 4 inactive
# total moderate-equivalent minutes per week (vigorous minutes count double); 0 for non-exercisers
d["PA_minutes"] = np.where(d.PhysActivity == 0, 0.0, raw.PA1MIN_)
d.loc[d.PhysActivity.isna(), "PA_minutes"] = np.nan
d["PA_minutes"] = d.PA_minutes.clip(upper=3000)                          # cap implausible >50 h/week
d["MainSport"] = np.where(d.PhysActivity == 0, 0, raw.EXRACT11)           # 0 = does not exercise
d.loc[d.PhysActivity.isna() | raw.EXRACT11.isin([77, 99]), "MainSport"] = np.nan
d.loc[(d.PhysActivity == 1) & raw.EXRACT11.isna(), "MainSport"] = np.nan
d["MainSport_MET"] = raw.METVL11_ / 10.0                                  # stored with 1 implied decimal

# muscle strengthening: 101-199 = times per week, 201-299 = times per month, 888 = never
s = raw.STRENGTH
per_week = np.where(s.between(101, 199), s - 100,
           np.where(s.between(201, 299), (s - 200) / 4.33,
           np.where(s == 888, 0, np.nan)))
d["Strength_per_week"] = np.clip(per_week, 0, 14)

# ---------- survey design ----------
d["weight"] = raw._LLCPWT
d["strata"] = raw._STSTR
d["psu"] = raw._PSU

STD21 = ["HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack",
         "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
         "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"]

n0 = len(d)
core = d.dropna(subset=["Diabetes"] + STD21).copy()
print(f"raw respondents: {n0:,}")
print(f"complete on outcome + 21 predictors: {len(core):,}")
core.to_csv("../data/analysis_core.csv.gz", index=False)

flow = pd.DataFrame({"step": ["Raw BRFSS 2015 respondents", "Complete outcome and 21 predictors"],
                     "n": [n0, len(core)]})
flow.to_csv("../results/sample_flow.csv", index=False)
