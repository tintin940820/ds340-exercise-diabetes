# saves the train / validation / test datasets as their own files in data/splits/
# same split the analysis scripts use: 70% train, 10% validation, 20% test, stratified, seed 42
# each row also gets a "dataset" column that says which set it is in

import os
import tools

os.makedirs("data/splits", exist_ok=True)

# Q1: BRFSS 2015, all adults, split on diabetes
q1 = tools.load_adults().drop(columns="w")
q1_parts = tools.split(q1, q1.Diabetes)

# Q2: BRFSS 2015 and 2023, ages 18-29, split on year and diabetes together
q2 = tools.load_young().drop(columns="w")
q2_parts = tools.split(q2, q2.year.astype(str) + "_" + q2.diabetes.astype(int).astype(str))

for q, parts, y in [("q1", q1_parts, "Diabetes"), ("q2", q2_parts, "diabetes")]:
    total = sum(len(p) for p in parts)
    for name, part in zip(["train", "validation", "test"], parts):
        part = part.copy()
        part.insert(0, "dataset", name)
        part.to_csv(f"data/splits/{q}_{name}.csv.gz", index=False)
        print(f"{q} {name:10s} {len(part):7d} rows  {100 * len(part) / total:4.1f}%  "
              f"diabetes {100 * part[y].mean():4.1f}%")
