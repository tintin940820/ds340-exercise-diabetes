# shared functions used by every script
# swap a block here and all scripts pick it up

from fractions import Fraction
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42

# the 21 features in the CDC Diabetes Health Indicators data (parent paper 1)
STD21 = ["HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack",
         "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
         "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"]


# ---- data

def load_adults():
    # BRFSS 2015, all adults
    d = pd.read_csv("data/analysis_core.csv.gz")
    d["w"] = d.weight / d.weight.mean()
    return d


def load_young():
    # BRFSS 2015 and 2023, ages 18-29, people who answered every habit question
    d = pd.read_csv("data/generation_core.csv.gz")
    d = d[d.year.isin([2015, 2023])]
    d = d.dropna(subset=["diabetes", "weight_status", "inactive", "current_smoker", "binge"])
    d["w"] = d.weight / d.weight.mean()
    return d


def split(d, strata, test=0.20, valid=0.10, seed=SEED):
    # stratified train / validation / test split (default 70 / 10 / 20)
    held = Fraction(str(test)) + Fraction(str(valid))
    train, rest = train_test_split(d, test_size=float(held), stratify=strata, random_state=seed)
    valid_part, test_part = train_test_split(rest, test_size=float(Fraction(str(test)) / held),
                                             stratify=strata[rest.index], random_state=seed)
    return train, valid_part, test_part


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


# ---- output

def barh(values, path, xlabel, highlight=None, ref=1, size=(6.5, 3.5)):
    fig, ax = plt.subplots(figsize=size)
    colors = ["#c4622d" if i == highlight else "#1f63a8" for i in values.index[::-1]]
    ax.barh(values.index[::-1], values[::-1], color=colors)
    ax.axvline(ref, color="#222222")
    ax.set_xlabel(xlabel)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
