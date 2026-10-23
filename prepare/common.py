"""Shared settings and helper functions used by every analysis step."""
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

# The robust covariance with survey weights was checked against a hand-computed sandwich
# estimator (identical to 6 decimals), so this statsmodels notice is silenced.
warnings.filterwarnings("ignore", message="cov_type not fully supported")

SEED = 42
BLUE, ORANGE, GRAY, INK = "#1f63a8", "#c4622d", "#8a8a8a", "#222222"

STD21 = ["HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack",
         "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
         "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"]

# covariate blocks for the step-wise logistic models
DEMOG = ["C(Age)", "Sex", "C(Income)", "C(Education)"]
LIFESTYLE = ["Smoker", "HvyAlcoholConsump", "Fruits", "Veggies", "AnyHealthcare", "NoDocbcCost", "CholCheck"]
CLINICAL = ["HighBP", "HighChol", "HeartDiseaseorAttack", "Stroke", "C(GenHlth)", "PhysHlth",
            "MentHlth", "DiffWalk"]

SPORT_NAMES = {
    0: "No exercise", 2: "Aerobics class/video", 6: "Exercise bike", 7: "Bicycling",
    11: "Calisthenics", 14: "Dancing", 15: "Elliptical machine", 18: "Gardening",
    19: "Golf (with cart)", 20: "Golf (walking)", 22: "Hiking", 28: "Jogging", 31: "Mowing lawn",
    37: "Running", 48: "Snow shoveling", 54: "Stair climbing", 57: "Swimming (leisure)",
    58: "Swimming laps", 61: "Tennis", 64: "Walking", 67: "Weight lifting", 69: "Yoga",
    72: "Farm/ranch work", 73: "Household activities", 76: "Yard work", 5: "Basketball",
    999: "Other activities"}


def load_core():
    return pd.read_csv("../data/analysis_core.csv.gz")


def plain_style():
    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 9.5, "axes.titlesize": 10.5,
        "axes.labelsize": 9.5, "axes.edgecolor": "#444444", "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "xtick.color": INK,
        "ytick.color": INK, "text.color": INK, "axes.labelcolor": INK,
        "axes.grid": False, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "legend.frameon": False})


def weighted_logit(formula, data):
    """Survey-weighted logistic regression.

    Weights are the BRFSS final weights rescaled to average 1. Standard errors are
    cluster-robust on the primary sampling unit within strata, a standard
    approximation to the complex-survey (Taylor linearisation) variance."""
    w = data["weight"] / data["weight"].mean()
    groups = pd.factorize(data["strata"].astype(str) + "_" + data["psu"].astype(str))[0]
    model = sm.GLM.from_formula(formula, data=data, family=sm.families.Binomial(), var_weights=w)
    return model.fit(cov_type="cluster", cov_kwds={"groups": groups})


def odds_ratios(fit, terms=None):
    ci = fit.conf_int()
    out = pd.DataFrame({"OR": np.exp(fit.params), "lo": np.exp(ci[0]), "hi": np.exp(ci[1]),
                        "p": fit.pvalues})
    if terms is not None:
        out = out.loc[[t for t in terms if t in out.index]]
    return out


def weighted_mean(x, w):
    m = x.notna()
    return float(np.average(x[m], weights=w[m]))
