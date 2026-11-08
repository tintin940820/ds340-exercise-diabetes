# reproduce parent paper 1 (Aich et al. 2025, arXiv 2505.22554), Table 3 on the CDC data
# same 253,680 adults and 21 features, same 4 models, all features vs Gumbel copula top-10
# paper setup: 80/20 stratified split, 20% of train held out to pick the threshold

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, roc_auc_score
from xgboost import XGBClassifier
import tools

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


def gumbel_scores(X, y):
    # paper 1: tau > 0 -> lambda_U = 2 - 2^(1/theta), theta = 1/(1-tau); tau <= 0 -> 0
    rows = []
    for c in X.columns:
        tau = kendalltau(X[c], y).statistic
        lam = 2 - 2 ** (1 - tau) if tau > 0 else 0.0
        rows.append({"feature": c, "tau": tau, "lambda_U": lam})
    return pd.DataFrame(rows).sort_values("lambda_U", ascending=False).reset_index(drop=True)


def make_model(name, y):
    pos_weight = (y == 0).sum() / (y == 1).sum()
    if name == "LR":
        return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
    if name == "RF":
        base = RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=tools.SEED)
    elif name == "GB":
        base = GradientBoostingClassifier(random_state=tools.SEED)
    else:
        base = XGBClassifier(scale_pos_weight=pos_weight, n_jobs=-1, random_state=tools.SEED)
    return CalibratedClassifierCV(base, method="isotonic", cv=5)


def fit(name, model, X, y):
    if name == "GB":
        # per-sample weights = inverse class frequency
        w = np.where(y == 1, len(y) / (2 * (y == 1).sum()), len(y) / (2 * (y == 0).sum()))
        return model.fit(X, y, sample_weight=w)
    return model.fit(X, y)


def best_threshold(p, y):
    # scan a fine grid of probability quantiles, keep the one with the best F1
    grid = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 197)))
    return max(grid, key=lambda t: f1_score(y, p >= t))


def evaluate(p, y, t, boot=200):
    pred = (p >= t).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, average="weighted", zero_division=0)
    rng = np.random.default_rng(tools.SEED)
    aucs = [roc_auc_score(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(boot))]
    return {"acc": accuracy_score(y, pred), "prec": prec, "rec": rec, "f1": f1,
            "auc": roc_auc_score(y, p), "auc_low": np.percentile(aucs, 2.5), "auc_high": np.percentile(aucs, 97.5)}


d = tools.load_adults()
print(len(d), "adults,", round(100 * d.Diabetes.mean(), 1), "% with diabetes or prediabetes")

train, valid, test = tools.split(d, d.Diabetes, test=0.20, valid=0.16)
print("train", len(train), "valid", len(valid), "test", len(test))

# copula feature screen on the training set
g = gumbel_scores(train[tools.STD21], train.Diabetes)
top10 = list(g.feature[:10])
print("\nGumbel copula scores (train):")
print(g.round(4).to_string(index=False))
print("\nour top 10:  ", top10)
print("paper top 10:", PAPER_TOP10)
print("same features:", sorted(top10) == sorted(PAPER_TOP10))
g.to_csv("results/parent_paper_1_gumbel_scores.csv", index=False)

rows = []
for set_name, cols in [("All", tools.STD21), ("Gumbel", top10)]:
    for name in ["LR", "RF", "GB", "XGB"]:
        y_tr = train.Diabetes.values
        m = fit(name, make_model(name, y_tr), train[cols], y_tr)
        t = best_threshold(m.predict_proba(valid[cols])[:, 1], valid.Diabetes.values)
        r = evaluate(m.predict_proba(test[cols])[:, 1], test.Diabetes.values, t)
        paper = dict(zip(["acc", "prec", "rec", "f1", "auc"], PAPER[(set_name, name)]))
        rows.append({"features": set_name, "model": name, **r, **{f"paper_{k}": v for k, v in paper.items()}})
        print(f"{set_name:6s} {name:3s}  ours auc {r['auc']:.3f} acc {r['acc']:.3f} f1 {r['f1']:.3f}"
              f"   paper auc {paper['auc']:.3f} acc {paper['acc']:.3f} f1 {paper['f1']:.3f}", flush=True)

out = pd.DataFrame(rows)
out["auc_diff"] = out.auc - out.paper_auc
out.to_csv("results/parent_paper_1_reproduction.csv", index=False)
print("\nours vs paper (test set):")
print(out[["features", "model", "auc", "paper_auc", "auc_diff", "acc", "paper_acc", "f1", "paper_f1"]]
      .round(3).to_string(index=False))
