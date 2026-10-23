"""Step 7. Build the generation dataset: young adults (age 18-29) from BRFSS 2015, 2021 and 2023.

Each raw file is read in chunks and only young adults and the needed columns are kept.
Variable names differ slightly by year, so each year has its own name map to one shared set of
names. Habits that are not asked in a year are left empty for that year (never filled in).
"""
import sys
import numpy as np
import pandas as pd

FILES = {2015: sys.argv[1], 2021: sys.argv[2], 2023: sys.argv[3]}

# shared name -> raw name in each year (None = not asked that year)
MAP = {
    "diab_raw":   {2015: "DIABETE3", 2021: "DIABETE4", 2023: "DIABETE4"},
    "exercise":   {2015: "_TOTINDA", 2021: "_TOTINDA", 2023: "_TOTINDA"},
    "bmi":        {2015: "_BMI5", 2021: "_BMI5", 2023: "_BMI5"},
    "smoke100":   {2015: "SMOKE100", 2021: "SMOKE100", 2023: "SMOKE100"},
    "smoker4":    {2015: "_SMOKER3", 2021: "_SMOKER3", 2023: "_SMOKER3"},
    "heavy_drink": {2015: "_RFDRHV5", 2021: "_RFDRHV7", 2023: "_RFDRHV8"},
    "binge":      {2015: "_RFBING5", 2021: "_RFBING5", 2023: "_RFBING6"},
    "fruit_lt1":  {2015: "_FRTLT1", 2021: "_FRTLT1A", 2023: None},
    "veg_lt1":    {2015: "_VEGLT1", 2021: "_VEGLT1A", 2023: None},
    "diffwalk":   {2015: "DIFFWALK", 2021: "DIFFWALK", 2023: "DIFFWALK"},
    "sex":        {2015: "SEX", 2021: "_SEX", 2023: "_SEX"},
    "educ":       {2015: "EDUCA", 2021: "EDUCA", 2023: "EDUCA"},
    "income":     {2015: "INCOME2", 2021: "INCOME3", 2023: "INCOME3"},
    "genhlth":    {2015: "GENHLTH", 2021: "GENHLTH", 2023: "GENHLTH"},
    "ecig":       {2015: None, 2021: "ECIGNOW1", 2023: "ECIGNOW2"},
    "age5":       {2015: "_AGEG5YR", 2021: "_AGEG5YR", 2023: "_AGEG5YR"},
    "weight":     {2015: "_LLCPWT", 2021: "_LLCPWT", 2023: "_LLCPWT"},
    "strata":     {2015: "_STSTR", 2021: "_STSTR", 2023: "_STSTR"},
    "psu":        {2015: "_PSU", 2021: "_PSU", 2023: "_PSU"},
}
parts = []
for year, path in FILES.items():
    keep = [v[year] for v in MAP.values() if v[year]]
    got = []
    for ch in pd.read_sas(path, format="xport", encoding="latin-1", chunksize=25000):
        ch = ch[ch[MAP["age5"][year]].between(1, 2)]          # 1 = 18-24, 2 = 25-29
        got.append(ch[keep])
    d = pd.concat(got, ignore_index=True)
    out = pd.DataFrame({"year": year}, index=d.index)
    for k, v in MAP.items():
        out[k] = d[v[year]] if v[year] else np.nan
    parts.append(out)
    print(year, len(out), flush=True)
pd.concat(parts, ignore_index=True).to_csv("../data/generation_raw_young.csv.gz", index=False)
print("saved")
