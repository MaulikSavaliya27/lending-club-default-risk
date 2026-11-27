"""Baseline default-prediction models for the Lending Club project.

Shared by train_model.py (evaluation) and app.py (risk scorer), so the
feature definitions live in exactly one place.

Two feature sets are compared on purpose:

* ``borrower``: what the borrower brings (income, FICO, DTI, credit history,
  loan size, term, purpose ...). No Lending Club grade, sub-grade, interest
  rate or installment. Grade and rate are Lending Club's own risk output, and
  installment is a function of loan amount, rate and term, so it leaks the rate.
* ``full``: adds grade, interest rate and installment. Shows how much of the
  signal is simply Lending Club's own underwriting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "default"
GRADES = list("ABCDEFG")

NUM_BORROWER = [
    "loan_amnt", "term", "dti", "delinq_2yrs", "open_acc", "pub_rec",
    "log_revol_bal", "revol_util", "total_acc", "emp_years", "emp_missing",
    "fico_avg", "log_income", "loan_to_income",
]
CAT_BORROWER = ["home_ownership", "verification_status", "purpose"]

NUM_LC = ["int_rate", "installment", "payment_to_income"]  # added in the "full" set
CAT_LC = ["grade"]

FEATURE_SETS = {
    "borrower": (NUM_BORROWER, CAT_BORROWER),
    # diagnostic only: borrower set without `term`, whose effect is inflated by censoring
    "borrower_noterm": ([c for c in NUM_BORROWER if c != "term"], CAT_BORROWER),
    "full": (NUM_BORROWER + NUM_LC, CAT_BORROWER + CAT_LC),
}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derived columns used by the models (pure function, no leakage of the target)."""
    out = df.copy()
    inc = out["annual_inc_capped"].clip(lower=1)
    out["log_income"] = np.log(inc)
    out["log_revol_bal"] = np.log1p(out["revol_bal"])
    out["loan_to_income"] = out["loan_amnt"] / inc
    out["payment_to_income"] = out["installment"] * 12 / inc
    out["home_ownership"] = out["home_ownership"].replace({"ANY": "OTHER"})
    return out


def build_pipeline(feature_set: str = "borrower", model: str = "logit") -> Pipeline:
    num, cat = FEATURE_SETS[feature_set]
    if model == "logit":
        pre = ColumnTransformer([
            ("num", StandardScaler(), num),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), cat),
        ])
        clf = LogisticRegression(C=0.5, max_iter=2000)
    elif model == "gbm":
        pre = ColumnTransformer([
            ("num", "passthrough", num),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=False), cat),
        ])
        clf = HistGradientBoostingClassifier(
            learning_rate=0.06, max_iter=250, max_leaf_nodes=15,
            min_samples_leaf=80, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15, random_state=42,
        )
    else:
        raise ValueError(model)
    return Pipeline([("pre", pre), ("clf", clf)])


def feature_columns(feature_set: str) -> list[str]:
    num, cat = FEATURE_SETS[feature_set]
    return num + cat


def ks_stat(y, p) -> float:
    """Kolmogorov-Smirnov separation between defaulters and non-defaulters."""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)}).sort_values("p")
    cum_bad = (d["y"].cumsum() / d["y"].sum()).to_numpy()
    cum_good = ((1 - d["y"]).cumsum() / (1 - d["y"]).sum()).to_numpy()
    return float(np.max(np.abs(cum_bad - cum_good)))


def evaluate(y, p) -> dict:
    y = np.asarray(y)
    p = np.asarray(p)
    top = p >= np.quantile(p, 0.9)
    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "ks": round(ks_stat(y, p), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "top_decile_default_rate": round(float(y[top].mean()), 4),
        "top_decile_lift": round(float(y[top].mean() / y.mean()), 2),
    }


def grade_only_scores(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Benchmark: score each loan by the historical default rate of its grade."""
    rates = train.groupby("grade", observed=True)[TARGET].mean()
    return test["grade"].astype(str).map(rates.rename(index=str)).fillna(train[TARGET].mean()).to_numpy()
