# reproduce parent paper 1 (Aich et al. 2025, arXiv 2505.22554), Table 3 on the CDC data
# same 253,680 adults and 21 features, same 4 models, all features vs Gumbel copula top-10
# paper setup: 80/20 stratified split, part of the 80% held out to pick the threshold.
# we use the project split (make_splits.py): fit on train (70%), pick the threshold on
# validation (10%), and score once on the unseen test set (20%, same size as the paper's test set).

import pandas as pd
import tools
from tools import gumbel_scores, make_model, fit, best_threshold, evaluate

# numbers reported in the paper (Table 3): acc, prec, rec, f1, auc
PAPER = {("All", "GB"): (0.804, 0.854, 0.804, 0.822, 0.827),
         ("All", "LR"): (0.791, 0.852, 0.791, 0.813, 0.820),
         ("All", "RF"): (0.779, 0.847, 0.779, 0.804, 0.801),
         ("All", "XGB"): (0.809, 0.853, 0.809, 0.826, 0.827),
         ("Gumbel", "GB"): (0.813, 0.850, 0.813, 0.827, 0.823),
         ("Gumbel", "LR"): (0.797, 0.849, 0.797, 0.817, 0.816),
         ("Gumbel", "RF"): (0.732, 0.826, 0.732, 0.766, 0.741),
         ("Gumbel", "XGB"): (0.810, 0.851, 0.810, 0.826, 0.822)}
PAPER_TOP10 = ["GenHlth", "HighBP", "DiffWalk", "HighChol", "BMI", "HeartDiseaseorAttack",
               "Age", "PhysHlth", "Stroke", "CholCheck"]


train, valid = tools.load_train_valid("q1")
print(round(100 * train.Diabetes.mean(), 1), "% with diabetes or prediabetes in train")

# copula feature screen on the training set
g = gumbel_scores(train[tools.STD21], train.Diabetes)
top10 = list(g.feature[:10])
print("\nGumbel copula scores (train):")
print(g.round(4).to_string(index=False))
print("\nour top 10:  ", top10)
print("paper top 10:", PAPER_TOP10)
print("same features:", sorted(top10) == sorted(PAPER_TOP10))
g.to_csv("results/parent_paper_1_gumbel_scores.csv", index=False)

# fit every model on train and pick its threshold on validation
y_tr = train.Diabetes.values
picked = []
for set_name, cols in [("All", tools.STD21), ("Gumbel", top10)]:
    for name in ["LR", "RF", "GB", "XGB"]:
        m = fit(name, make_model(name, y_tr), train[cols], y_tr)
        t = best_threshold(m.predict_proba(valid[cols])[:, 1], valid.Diabetes.values)
        picked.append((set_name, name, cols, m, t))
        print(f"fit {set_name:6s} {name:3s}  threshold {t:.3f}", flush=True)

# everything is picked, now the test set is used once
test = tools.load_test("q1")
rows = []
for set_name, name, cols, m, t in picked:
    r = evaluate(m.predict_proba(test[cols])[:, 1], test.Diabetes.values, t)
    paper = dict(zip(["acc", "prec", "rec", "f1", "auc"], PAPER[(set_name, name)]))
    rows.append({"features": set_name, "model": name, **r, **{f"paper_{k}": v for k, v in paper.items()}})
    print(f"{set_name:6s} {name:3s}  ours auc {r['auc']:.3f} acc {r['acc']:.3f} f1 {r['f1']:.3f}"
          f"   paper auc {paper['auc']:.3f} acc {paper['acc']:.3f} f1 {paper['f1']:.3f}")

out = pd.DataFrame(rows)
out["auc_diff"] = out.auc - out.paper_auc
out.to_csv("results/parent_paper_1_reproduction.csv", index=False)
print("\nours vs paper (test set):")
print(out[["features", "model", "auc", "paper_auc", "auc_diff", "acc", "paper_acc", "f1", "paper_f1"]]
      .round(3).to_string(index=False))
