# shared functions used by every script
# swap a block here and all scripts pick it up

from fractions import Fraction
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import kendalltau
from sklearn.model_selection import train_test_split
from sklearn.metrics import (log_loss, roc_auc_score, accuracy_score, f1_score,
                             precision_recall_fscore_support)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import mutual_info_classif
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42

# the 21 features in the CDC Diabetes Health Indicators data (parent paper 1)
STD21 = ["HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack",
         "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
         "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"]


# ---- data
# the project uses ONE split, made once by make_splits.py and saved in data/splits/:
#   train 70%       fit the models
#   validation 10%  pick the model / the threshold / the features
#   test 20%        unseen: opened once, at the very end, after everything is picked
# every analysis script reads these files. no script makes its own split.

SPLIT_DIR = "data/splits"
PARTS = ["train", "validation", "test"]


def load_adults():
    # BRFSS 2015, all adults (Q1 and parent paper 1). only make_splits.py calls this
    d = pd.read_csv("data/analysis_core.csv.gz")
    d.insert(0, "row_id", np.arange(len(d)))
    return d


def load_young():
    # BRFSS 2015 and 2023, ages 18-29, people who answered every habit question (Q2).
    # only make_splits.py calls this
    d = pd.read_csv("data/generation_core.csv.gz")
    d.insert(0, "row_id", np.arange(len(d)))
    d = d[d.year.isin([2015, 2023])]
    return d.dropna(subset=["diabetes", "weight_status", "inactive", "current_smoker", "binge"])


def split(d, strata, test=0.20, valid=0.10, seed=SEED):
    # stratified train / validation / test split (70 / 10 / 20). only make_splits.py calls this
    held = Fraction(str(test)) + Fraction(str(valid))
    train, rest = train_test_split(d, test_size=float(held), stratify=strata, random_state=seed)
    valid_part, test_part = train_test_split(rest, test_size=float(Fraction(str(test)) / held),
                                             stratify=strata[rest.index], random_state=seed)
    return train, valid_part, test_part


def _load(q, part):
    d = pd.read_csv(f"{SPLIT_DIR}/{q}_{part}.csv.gz")
    if not (d.dataset == part).all():
        raise ValueError(f"{q}_{part} file has rows labeled as another set")
    d["w"] = d.weight / d.weight.mean()  # survey weight, scaled to mean 1
    return d


def load_train_valid(q):
    # q = "q1" (2015 adults) or "q2" (ages 18-29, 2015 and 2023)
    train, valid = _load(q, "train"), _load(q, "validation")
    print(f"{q}: train {len(train)} rows, validation {len(valid)} rows (test set not opened yet)")
    return train, valid


def load_test(q):
    # the unseen set. call this only after the model is picked on train + validation
    test = _load(q, "test")
    print(f"\n--- opening the {q} test set ({len(test)} rows), used once ---")
    return test


def wpct(x, w):
    # weighted percent
    return 100 * np.average(x, weights=w)


# ---- survey-weighted logistic regression

def fit_logit(formula, data):
    return smf.glm(formula, data=data, family=sm.families.Binomial(), var_weights=data.w).fit()


def score(model, data, y, name, part):
    p = model.predict(data)
    return {"model": name, "set": part, "log_loss": log_loss(data[y], p, sample_weight=data.w),
            "auc": roc_auc_score(data[y], p, sample_weight=data.w)}


def fit_and_pick(formulas, train, valid, y):
    # fit every candidate on train, keep the one with the lowest validation log loss
    fits = {name: fit_logit(f, train) for name, f in formulas.items()}
    scores = [score(m, valid, y, name, "valid") for name, m in fits.items()]
    best = min(scores, key=lambda s: s["log_loss"])["model"]
    return fits, scores, best


def odds_ratios(model, rows):
    ci = np.exp(model.conf_int().loc[rows])
    return pd.DataFrame({"odds_ratio": np.exp(model.params[rows]), "low_95": ci[0], "high_95": ci[1]})


# ---- parent paper 1 models (classifiers, threshold, scoring)

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
        base = RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=SEED)
    elif name == "GB":
        base = GradientBoostingClassifier(random_state=SEED)
    else:
        base = XGBClassifier(scale_pos_weight=pos_weight, n_jobs=-1, random_state=SEED)
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
    rng = np.random.default_rng(SEED)
    aucs = [roc_auc_score(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(boot))]
    return {"acc": accuracy_score(y, pred), "prec": prec, "rec": rec, "f1": f1,
            "auc": roc_auc_score(y, p), "auc_low": np.percentile(aucs, 2.5), "auc_high": np.percentile(aucs, 97.5)}


# ---- feature selection block (swap the method, everything else stays the same)

# extra exercise detail we recovered from the raw 2015 file (not in the public CDC table)
EXERCISE = ["PA_minutes", "MainSport_MET", "Strength_per_week", "PA_level"]


def two_sided_gumbel_scores(X, y):
    # our change: a feature that goes DOWN with diabetes (tau < 0) is flipped (-x),
    # which is the 180-degree rotated Gumbel copula, so it gets the same score as a
    # feature that goes up by the same amount: lambda = 2 - 2^(1 - |tau|)
    rows = []
    for c in X.columns:
        tau = kendalltau(X[c], y).statistic
        rows.append({"feature": c, "tau": tau, "lambda_U": 2 - 2 ** (1 - abs(tau)),
                     "direction": "protective" if tau < 0 else "risk"})
    return pd.DataFrame(rows).sort_values("lambda_U", ascending=False).reset_index(drop=True)


def select_features(method, X, y, k=10):
    # returns the top k columns of X picked by one method
    if method == "gumbel":
        return list(gumbel_scores(X, y).feature[:k])
    if method == "two_sided_gumbel":
        return list(two_sided_gumbel_scores(X, y).feature[:k])
    if method == "mutual_info":
        sub = X.sample(min(len(X), 50000), random_state=SEED)
        mi = mutual_info_classif(sub, y[sub.index], random_state=SEED)
        return list(pd.Series(mi, X.columns).sort_values(ascending=False).index[:k])
    if method == "l1_logistic":
        # smallest penalty that keeps about k features, ranked by size of the coefficient
        Xs = StandardScaler().fit_transform(X)
        coef = LogisticRegression(penalty="l1", solver="liblinear", C=0.01).fit(Xs, y).coef_[0]
        return list(pd.Series(np.abs(coef), X.columns).sort_values(ascending=False).index[:k])
    raise ValueError(method)


# ---- output

def barh(values, path, xlabel, highlight=None, ref=1, size=(6.5, 3.5)):
    fig, ax = plt.subplots(figsize=size)
    colors = ["#c4622d" if i == highlight else "#1f63a8" for i in values.index[::-1]]
    ax.barh(values.index[::-1], values[::-1], color=colors)
    ax.axvline(ref, color="#222222")
    ax.set_xlabel(xlabel)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
