"""Train and evaluate the baseline default models.

Run:  python train_model.py
Writes: models/metrics.json, models/deciles.csv, models/coefficients.csv,
        models/numeric_effects.csv and images/model_*.png

Two validation schemes:
  * random - stratified 80/20 split over all loans (optimistic: same vintages
             in train and test).
  * mature - out-of-time: train on 36-month loans issued 2013-14, test on
             36-month loans issued 2015. Only 36-month loans issued through 2015
             have finished by the end of the data (2018Q4), so every outcome is
             known. Testing on 2016 loans, or on 60-month loans issued after
             2013, would be biased: the file keeps only CLOSED loans, so
             still-running (mostly good) loans are missing and defaults look
             too frequent (right-censoring).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

from risk_model import TARGET, add_features, build_pipeline, evaluate, feature_columns, grade_only_scores

ROOT = Path(__file__).parent
OUT = ROOT / "models"
IMG = ROOT / "images"
OUT.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")

df = add_features(pd.read_csv(ROOT / "data" / "processed" / "loans_clean.csv"))

splits = {}
tr, te = train_test_split(df, test_size=0.2, stratify=df[TARGET], random_state=42)
splits["random"] = (tr, te)
d36 = df[df.term == 36]
splits["mature"] = (d36[d36.issue_year <= 2014], d36[d36.issue_year == 2015])

CONFIGS = [("borrower", "logit"), ("borrower", "gbm"), ("full", "logit"), ("full", "gbm")]


def auc_ci(y, p, n=300, seed=0):
    """95% bootstrap interval for ROC AUC."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        vals.append(roc_auc_score(y[i], p[i]))
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


metrics, scores = {}, {}
for split, (train, test) in splits.items():
    metrics[split] = {"n_train": len(train), "n_test": len(test),
                      "test_default_rate": round(float(test[TARGET].mean()), 4), "models": {}}
    g = grade_only_scores(train, test)
    metrics[split]["models"]["grade_only (benchmark)"] = {**evaluate(test[TARGET], g), "roc_auc_ci95": auc_ci(test[TARGET], g)}
    for fs, m in CONFIGS:
        pipe = build_pipeline(fs, m).fit(train[feature_columns(fs)], train[TARGET])
        p = pipe.predict_proba(test[feature_columns(fs)])[:, 1]
        scores[(split, fs, m)] = (test[TARGET].to_numpy(), p)
        metrics[split]["models"][f"{fs}_{m}"] = {**evaluate(test[TARGET], p), "roc_auc_ci95": auc_ci(test[TARGET], p)}

# ---- diagnostic: is the "2016 drift" real or censoring? 36-month loans, train on earlier years ----
from sklearn.metrics import roc_auc_score
diag = []
for test_year in (2014, 2015, 2016):
    trn, tst = d36[d36.issue_year < test_year], d36[d36.issue_year == test_year]
    pp = build_pipeline("borrower", "logit").fit(trn[feature_columns("borrower")], trn[TARGET]).predict_proba(tst[feature_columns("borrower")])[:, 1]
    diag.append({"test_year": test_year, "n_test": len(tst), "actual": round(float(tst[TARGET].mean()), 4),
                 "predicted": round(float(pp.mean()), 4), "gap_pts": round(float((tst[TARGET].mean() - pp.mean()) * 100), 1),
                 "auc": round(float(roc_auc_score(tst[TARGET], pp)), 4),
                 "outcomes_complete": bool(test_year <= 2015)})
metrics["calibration_by_test_year_36m"] = diag

# ---- diagnostic: how much of the borrower model's random-split AUC comes from `term`? ----
tr_r, te_r = splits["random"]
p_nt = build_pipeline("borrower_noterm", "logit").fit(tr_r[feature_columns("borrower_noterm")], tr_r[TARGET]).predict_proba(te_r[feature_columns("borrower_noterm")])[:, 1]
metrics["diagnostics"] = {"random_split_borrower_logit_without_term_auc": round(float(roc_auc_score(te_r[TARGET], p_nt)), 4)}
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

# ---- decile table: borrower-only logit, mature out-of-time test ----
y, p = scores[("mature", "borrower", "logit")]
d = pd.DataFrame({"y": y, "p": p})
d["decile"] = pd.qcut(d["p"].rank(method="first"), 10, labels=range(1, 11))
dec = d.groupby("decile", observed=True).agg(loans=("y", "size"), predicted=("p", "mean"), actual=("y", "mean")).round(4)
dec.to_csv(OUT / "deciles.csv")

# ---- coefficients: borrower-only logit fit on all data ----
pipe = build_pipeline("borrower", "logit").fit(df[feature_columns("borrower")], df[TARGET])
names = pipe.named_steps["pre"].get_feature_names_out()
coef = pd.DataFrame({"feature": [n.split("__", 1)[1] for n in names], "coef": pipe.named_steps["clf"].coef_[0]})
coef["abs"] = coef["coef"].abs()
coef = coef.sort_values("abs", ascending=False).drop(columns="abs").round(4)
coef.to_csv(OUT / "coefficients.csv", index=False)

# ---- charts ----
labels = {"borrower_logit": "Borrower-only logistic", "full_logit": "Full (with grade, rate) logistic",
          "borrower_gbm": "Borrower-only GBM", "full_gbm": "Full GBM"}
fig, ax = plt.subplots(figsize=(7, 5.5))
for (split, fs, m), (yy, pp) in scores.items():
    if split == "mature" and m == "logit":
        fpr, tpr, _ = roc_curve(yy, pp)
        ax.plot(fpr, tpr, lw=2, label=f"{labels[f'{fs}_{m}']} (AUC {metrics[split]['models'][f'{fs}_{m}']['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess")
ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC: train 2013-14, test 2015 (36-month loans)")
ax.title.set_fontweight("bold")
ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig(IMG / "model_roc.png", dpi=130); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
x = np.arange(len(dec))
axes[0].bar(x, dec["actual"] * 100, color=sns.color_palette("YlOrRd", 10))
for i, v in enumerate(dec["actual"] * 100):
    axes[0].text(i, v + 0.6, f"{v:.0f}%", ha="center", fontsize=9, fontweight="bold")
axes[0].set_xticks(x, dec.index.astype(str))
axes[0].set(xlabel="Predicted-risk decile (1 = safest)", ylabel="Actual default rate (%)", title="Default rate by risk decile")
yy, pp = scores[("mature", "borrower", "logit")]
frac, mean_p = calibration_curve(yy, pp, n_bins=10, strategy="quantile")
axes[1].plot(mean_p, frac, "o-", lw=2, label="Borrower-only logistic")
axes[1].plot([0, 0.45], [0, 0.45], "k--", lw=1, label="Perfect calibration")
axes[1].set(xlabel="Predicted default probability", ylabel="Observed default rate", title="Calibration (2015, 36-month loans)")
axes[1].legend()
for a in axes: a.title.set_fontweight("bold")
fig.tight_layout(); fig.savefig(IMG / "model_deciles_calibration.png", dpi=130); plt.close(fig)

from risk_model import NUM_BORROWER
num = coef[coef["feature"].isin(NUM_BORROWER)].copy()
num["odds_change_pct"] = (np.exp(num["coef"]) - 1) * 100
num = num.sort_values("odds_change_pct")
num.to_csv(OUT / "numeric_effects.csv", index=False)
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.barh(num["feature"], num["odds_change_pct"], color=["#b2182b" if c > 0 else "#2166ac" for c in num["odds_change_pct"]])
ax.axvline(0, color="k", lw=1)
ax.set(xlabel="Change in default odds per +1 standard deviation (%)", title="Numeric drivers in the borrower-only model")
ax.title.set_fontweight("bold")
fig.text(0.99, 0.01, "red = raises default risk, blue = lowers it. open_acc and total_acc are correlated.", ha="right", fontsize=8, color="gray")
fig.tight_layout(); fig.savefig(IMG / "model_drivers.png", dpi=130); plt.close(fig)

print(json.dumps({s: {k: v["models"][k] for k in v["models"]} for s, v in metrics.items() if s in splits}, indent=1))
print(dec); print(coef.head(14))
