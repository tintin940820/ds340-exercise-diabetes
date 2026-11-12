# step 1 of the project: split each dataset ONE time into train / validation / test
#   train 70%       fit the models
#   validation 10%  pick the model, the threshold and the features
#   test 20%        unseen: no script opens it until its model is picked, then it is used once
# stratified (same diabetes rate in every set), seed 42, so anyone who runs this gets the same rows.
# every row keeps its row_id (row number in the clean table) and gets a "dataset" label.
# the checks at the bottom stop with an error if anything is wrong.

import os
import pandas as pd
import tools

os.makedirs(tools.SPLIT_DIR, exist_ok=True)

q1 = tools.load_adults()      # Q1 and parent paper 1: BRFSS 2015, all adults
q2 = tools.load_young()       # Q2: BRFSS 2015 and 2023, ages 18-29
datasets = {
    # Q1 is split on diabetes, Q2 on year and diabetes together (so 2015 and 2023 are balanced too)
    "q1": (q1, "Diabetes", tools.split(q1, q1.Diabetes)),
    "q2": (q2, "diabetes", tools.split(q2, q2.year.astype(str) + "_" + q2.diabetes.astype(int).astype(str))),
}

for q, (full, y, parts) in datasets.items():
    print(f"\n{q}: {len(full)} rows")
    for name, part in zip(tools.PARTS, parts):
        part = part.copy()
        part.insert(0, "dataset", name)
        # mtime 0 so the file is byte-for-byte the same every time it is rebuilt
        part.to_csv(f"{tools.SPLIT_DIR}/{q}_{name}.csv.gz", index=False,
                    compression={"method": "gzip", "mtime": 0})
        print(f"  {name:10s} {len(part):7d} rows  {100 * len(part) / len(full):4.1f}%  "
              f"diabetes {100 * part[y].mean():4.1f}%")

# ---- checks (read the files back from disk)
print("\nchecks:")
for q, (full, y, _) in datasets.items():
    saved = {p: pd.read_csv(f"{tools.SPLIT_DIR}/{q}_{p}.csv.gz") for p in tools.PARTS}
    ids = {p: set(d.row_id) for p, d in saved.items()}
    assert not ids["train"] & ids["validation"], "train and validation share rows"
    assert not ids["train"] & ids["test"], "train and test share rows"
    assert not ids["validation"] & ids["test"], "validation and test share rows"
    assert set.union(*ids.values()) == set(full.row_id), "some rows are missing"
    for p, share in zip(tools.PARTS, [0.70, 0.10, 0.20]):
        assert (saved[p].dataset == p).all(), f"{p} file has a wrong label"
        assert abs(len(saved[p]) / len(full) - share) < 0.001, f"{p} is not {share:.0%}"
        assert abs(saved[p][y].mean() - full[y].mean()) < 0.002, f"{p} diabetes rate is off"
    print(f"  {q}: no shared rows, every row used once, 70/10/20, same diabetes rate, labels ok")
