# our change to parent paper 1: swap the feature selection block
# paper 1 (Aich et al. 2025) gives every protective feature (tau <= 0) a score of zero,
# so exercise, diet, income and education can never be picked.
# we try 4 ways to pick the top 10 features, plus one that swaps the yes/no exercise answer
# for the exercise detail we recovered from the raw file (minutes, MET of main sport, strength days, level),
# and train the same models on each. same data, same split (data/splits), same models as the reproduction.

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tools
from tools import make_model, fit, best_threshold, evaluate, select_features

train, valid = tools.load_train_valid("q1")


def fix_met(d):
    # exercise detail: people with no exercise have no main sport, so their MET is 0
    d["MainSport_MET"] = d.MainSport_MET.where(d.PhysActivity == 1, 0)
    return d


train, valid = fix_met(train), fix_met(valid)
medians = train[tools.EXERCISE].median()  # a few missing answers -> training median
train, valid = train.fillna(medians), valid.fillna(medians)

y_tr = train.Diabetes
wide = tools.STD21 + tools.EXERCISE

# the two-sided scores, so we can see what changed
scores = tools.two_sided_gumbel_scores(train[wide], y_tr)
scores.to_csv("results/novelty_two_sided_scores.csv", index=False)
print("\ntwo-sided Gumbel scores (21 features + exercise detail):")
print(scores.round(4).to_string(index=False))

sets = {
    "All 21 (paper)": tools.STD21,
    "Gumbel top 10 (paper)": select_features("gumbel", train[tools.STD21], y_tr),
    "Two-sided Gumbel top 10 (ours)": select_features("two_sided_gumbel", train[tools.STD21], y_tr),
    "Mutual information top 10": select_features("mutual_info", train[tools.STD21], y_tr),
    "L1 logistic top 10": select_features("l1_logistic", train[tools.STD21], y_tr),
}
# second change: swap the yes/no exercise answer for the exercise detail we recovered
ours = sets["Two-sided Gumbel top 10 (ours)"]
sets["Two-sided top 10, yes/no exercise -> exercise detail (ours)"] = (
    [c for c in ours if c != "PhysActivity"] + tools.EXERCISE)
protective = set(scores.feature[scores.direction == "protective"])
print("\nfeatures picked:")
for name, cols in sets.items():
    print(f"  {name}: {cols}")

# fit every model on train and pick its threshold on validation
picked = []
for set_name, cols in sets.items():
    for name in ["LR", "GB", "XGB"]:
        m = fit(name, make_model(name, y_tr.values), train[cols], y_tr.values)
        t = best_threshold(m.predict_proba(valid[cols])[:, 1], valid.Diabetes.values)
        picked.append((set_name, name, cols, m, t))
        print(f"fit {set_name:50s} {name:3s} threshold {t:.3f}", flush=True)

# everything is picked, now the test set is used once
test = fix_met(tools.load_test("q1")).fillna(medians)
rows = []
for set_name, name, cols, m, t in picked:
    r = evaluate(m.predict_proba(test[cols])[:, 1], test.Diabetes.values, t)
    rows.append({"features": set_name, "model": name, "n_features": len(cols),
                 "protective_kept": len(protective & set(cols)), **r})
    print(f"{set_name:50s} {name:3s} auc {r['auc']:.3f} ({r['auc_low']:.3f}-{r['auc_high']:.3f})"
          f" acc {r['acc']:.3f} f1 {r['f1']:.3f}")

out = pd.DataFrame(rows)
out.to_csv("results/novelty_comparison.csv", index=False)
print("\ncomparison (test set):")
print(out[["features", "model", "protective_kept", "auc", "auc_low", "auc_high", "acc", "f1"]]
      .round(3).to_string(index=False))

# figure: test AUC for each feature set, one bar per model
pivot = out.pivot(index="features", columns="model", values="auc").loc[list(sets)][["LR", "GB", "XGB"]]
ax = pivot.plot.barh(figsize=(7.5, 4.2), color=["#1f63a8", "#c4622d", "#6a9a3a"])
ax.invert_yaxis()
ax.set_xlim(0.78, 0.84)
ax.set_xlabel("test AUC")
ax.set_ylabel("")
plt.savefig("figures/novelty_comparison.png", dpi=200, bbox_inches="tight")
