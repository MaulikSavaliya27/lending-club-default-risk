"""Lending Club Default Risk Dashboard.  Run: streamlit run app.py"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from risk_model import GRADES, add_features, build_pipeline, feature_columns

st.set_page_config(page_title="Lending Club Default Risk", page_icon="🏦", layout="wide")

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "processed" / "loans_clean.csv"
MODELS = ROOT / "models"
IMAGES = ROOT / "images"


@st.cache_data
def load(src):
    df = pd.read_csv(src)
    df["grade"] = pd.Categorical(df["grade"], categories=GRADES, ordered=True)
    return df


def default_rate(df, col):
    """Default rate (%) per category of `col`, indexed by plain strings."""
    s = df.groupby(col, observed=True)["default"].mean() * 100
    s.index = s.index.astype(str)
    return s.round(1)


def apply_filters(df, grades, terms, purposes, years, fico):
    return df[
        df["grade"].isin(grades)
        & df["term"].isin(terms)
        & df["purpose"].isin(purposes)
        & df["issue_year"].between(*years)
        & df["fico_avg"].between(*fico)
    ]


@st.cache_resource(show_spinner="Training the baseline model...")
def fit_scorer(n_rows, _df):
    """Borrower-only logistic regression on all loans + 5-fold out-of-fold scores."""
    d = add_features(_df)
    X, y = d[feature_columns("borrower")], d["default"]
    model = build_pipeline("borrower", "logit").fit(X, y)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    oof = cross_val_predict(build_pipeline("borrower", "logit"), X, y, cv=cv, method="predict_proba")[:, 1]
    return model, pd.Series(oof, index=_df.index, name="p_default")


@st.cache_data(show_spinner=False)
def run_tests(n_rows, _df):
    """Re-run the four notebook hypothesis tests on the full cleaned data."""
    d = _df
    paid, bad = d[d["default"] == 0], d[d["default"] == 1]
    t = stats.ttest_ind(paid["annual_inc"], bad["annual_inc"])
    welch = stats.ttest_ind(paid["annual_inc"], bad["annual_inc"], equal_var=False)
    mw = stats.mannwhitneyu(paid["annual_inc"], bad["annual_inc"])
    sp = np.sqrt(((len(paid) - 1) * paid["annual_inc"].var() + (len(bad) - 1) * bad["annual_inc"].var()) / (len(paid) + len(bad) - 2))
    cohen_d = (paid["annual_inc"].mean() - bad["annual_inc"].mean()) / sp
    d_label = "negligible" if abs(cohen_d) < 0.2 else "small" if abs(cohen_d) < 0.5 else "medium or larger"
    ct = pd.crosstab(d["purpose"], d["default"])
    chi2, p_chi, dof, _ = stats.chi2_contingency(ct)
    v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
    groups = [g["int_rate"].to_numpy() for _, g in d.groupby("grade", observed=True)]
    anova = stats.f_oneway(*groups)
    tuk = stats.tukey_hsd(*groups).pvalue
    ss_between = sum(len(g) * (g.mean() - d["int_rate"].mean()) ** 2 for g in groups)
    eta2 = ss_between / ((d["int_rate"] - d["int_rate"].mean()) ** 2).sum()
    n_pairs = len(groups) * (len(groups) - 1) // 2
    sig_pairs = int((tuk[np.triu_indices(len(groups), 1)] < 0.05).sum())
    a, b = d[d["term"] == 60]["default"], d[d["term"] == 36]["default"]
    # same-grade comparison: per-grade ratio and Mantel-Haenszel common odds ratio
    cell = d.groupby(["grade", "term"], observed=True)["default"].agg(["sum", "size"]).unstack("term")
    num = den = 0.0
    for _, c in cell.iterrows():
        a6, n6, a3, n3 = c[("sum", 60)], c[("size", 60)], c[("sum", 36)], c[("size", 36)]
        num += a6 * (n3 - a3) / (n6 + n3)
        den += (n6 - a6) * a3 / (n6 + n3)
    mh_or = num / den
    ratios = (cell[("sum", 60)] / cell[("size", 60)]) / (cell[("sum", 36)] / cell[("size", 36)])
    crude_or = (a.mean() / (1 - a.mean())) / (b.mean() / (1 - b.mean()))
    p = d["default"].mean()
    z = (a.mean() - b.mean()) / np.sqrt(p * (1 - p) * (1 / len(a) + 1 / len(b)))
    p_z = stats.norm.sf(z)  # right-tailed, as in the notebook
    p_z_txt = "p ≈ 0" if p_z < 1e-300 else f"p = {p_z:.1e}"
    return pd.DataFrame({
        "Question": ["Do defaulters earn less/more?", "Does default depend on purpose?",
                     "Do interest rates differ by grade?", "Do 60-month loans default more?"],
        "H0": ["Mean income of Fully Paid = mean income of Charged Off (μ₁ − μ₂ = 0)",
               "Default status and loan purpose are independent (no association)",
               "Mean interest rate is equal across all grades A-G (μA = μB = … = μG)",
               "Default proportion of 60-month loans ≤ that of 36-month loans (p₆₀ − p₃₆ ≤ 0)"],
        "H1": ["Mean income of Fully Paid ≠ mean income of Charged Off (two-tailed)",
               "Default status and loan purpose are associated (not independent)",
               "At least one grade has a different mean interest rate",
               "Default proportion of 60-month loans > that of 36-month loans (right-tailed)"],
        "Test": ["Two-sample t-test", "Chi-square + Cramér's V", "One-way ANOVA + Tukey", "Two-proportion z-test"],
        "Result": [f"t = {t.statistic:.2f}, p = {t.pvalue:.1e}",
                   f"χ² = {chi2:.1f}, dof = {dof}, V = {v:.3f}",
                   f"F = {anova.statistic:,.0f}; {sig_pairs} of {n_pairs} grade pairs differ",
                   f"z = {z:.1f}, {p_z_txt} ({a.mean():.1%} vs {b.mean():.1%})"],
        "Robustness check": [f"Effect size: Cohen's d = {cohen_d:.2f} ({d_label}). Welch t = {welch.statistic:.2f}; Mann-Whitney p = {mw.pvalue:.1e}",
                              "Weak effect size: read V, not p",
                              f"η² = {eta2:.2f}: grade is assigned from the rate, so this is near-tautological, not proof that pricing matches risk",
                              f"Not controlled for grade (60-month loans are mostly weaker grades). Within the same grade: {ratios.min():.1f}-{ratios.max():.1f}× "
                              f"(Mantel-Haenszel OR = {mh_or:.2f} vs crude {crude_or:.2f})"],
        "Decision": ["Reject H₀"] * 4,
    })


# ---------- data ----------
if DATA.exists():
    df = load(DATA)
else:
    up = st.file_uploader("Upload loans_clean.csv", type="csv")
    if up is None:
        st.info("`data/processed/loans_clean.csv` not found. Upload it to continue.")
        st.stop()
    df = load(up)

# ---------- sidebar filters ----------
st.sidebar.header("Filters")
st.sidebar.caption("Apply to every tab except Hypothesis tests and the Risk scorer.")
grades = st.sidebar.multiselect("Grade", GRADES, default=GRADES)
terms = st.sidebar.multiselect("Term (months)", [36, 60], default=[36, 60])
all_purposes = sorted(df["purpose"].unique())
purposes = st.sidebar.multiselect("Purpose", all_purposes, default=all_purposes)
years = st.sidebar.slider("Issue year", int(df["issue_year"].min()), int(df["issue_year"].max()),
                          (int(df["issue_year"].min()), int(df["issue_year"].max())))
fico_lo, fico_hi = int(np.floor(df["fico_avg"].min())), int(np.ceil(df["fico_avg"].max()))  # ceil: max is 847.5
fico = st.sidebar.slider("FICO score", fico_lo, fico_hi, (fico_lo, fico_hi))

f = apply_filters(df, grades, terms, purposes, years, fico)
if f.empty:
    st.warning("No loans match these filters.")
    st.stop()

st.sidebar.download_button("Download filtered loans (CSV)", f.to_csv(index=False).encode(),
                           "loans_filtered.csv", "text/csv")

# ---------- header + KPIs ----------
st.title("🏦 Lending Club Loan Default Risk")
st.caption(f"{len(df):,} closed loans, 2013-2016 · default = Charged Off")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Loans", f"{len(f):,}")
c2.metric("Default rate", f"{f['default'].mean():.1%}", f"{(f['default'].mean() - df['default'].mean()) * 100:+.1f} pts vs all", delta_color="inverse")
c3.metric("Avg interest rate", f"{f['int_rate'].mean():.1f}%")
c4.metric("Avg FICO", f"{f['fico_avg'].mean():.0f}")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Default drivers", "Grade × Term", "Credit signals", "Hypothesis tests",
     "Model performance", "Risk scorer", "Recommendations"])

with tab1:
    a, b = st.columns(2)
    a.subheader("Default rate (%) by grade")
    a.bar_chart(default_rate(f, "grade"))
    b.subheader("Default rate (%) by term")
    b.bar_chart(default_rate(f, "term"))
    a.subheader("Default rate (%) by purpose")
    big = f["purpose"].value_counts()
    a.bar_chart(default_rate(f[f["purpose"].isin(big[big >= 300].index)], "purpose"))
    a.caption("Purposes with fewer than 300 loans are hidden (too noisy).")
    b.subheader("Default rate (%) by verification")
    b.bar_chart(default_rate(f, "verification_status"))
    b.caption("Verified borrowers default more, probably because riskier applicants are the ones who get verified (not tested here).")
    b.subheader("Default rate (%) by issue year")
    b.line_chart(default_rate(f, "issue_year"))
    b.caption("Rising with vintage, partly a survivorship artefact: only loans already closed are "
              "included, and recent loans that closed early skew toward charge-offs.")

with tab2:
    pivot = f.pivot_table(values="default", index="grade", columns="term", aggfunc="mean", observed=True) * 100
    pivot.index = pivot.index.astype(str)
    pivot.columns = [f"{c} months" for c in pivot.columns]
    st.bar_chart(pivot.round(1))
    st.dataframe(pivot.round(1), width="stretch")
    if {"36 months", "60 months"} <= set(pivot.columns):
        ratio = (pivot["60 months"] / pivot["36 months"]).dropna()
        st.caption(f"60-month loans default more in every grade shown: {ratio.min():.1f}-{ratio.max():.1f}× the 36-month rate "
                   "within a grade, and the absolute gap is widest for grades D-E. The overall ~2× gap is inflated because "
                   "60-month loans are concentrated in weaker grades.")

with tab3:
    a, b = st.columns(2)
    fico_band = pd.cut(f["fico_avg"], bins=[600, 660, 680, 700, 720, 740, 760, 900],
                       labels=["600-660", "660-680", "680-700", "700-720", "720-740", "740-760", "760+"]).astype(str)
    a.subheader("Default rate (%) by FICO band")
    a.bar_chart(default_rate(f.assign(fico_band=fico_band), "fico_band"))
    rate_band = pd.cut(f["int_rate"], bins=[0, 8, 11, 14, 17, 20, 25, 40],
                       labels=["00-08%", "08-11%", "11-14%", "14-17%", "17-20%", "20-25%", "25-40%"]).astype(str)
    b.subheader("Default rate (%) by interest-rate band")
    b.bar_chart(default_rate(f.assign(rate_band=rate_band), "rate_band"))
    st.caption("Grade and interest rate are Lending Club's own risk outputs, so they summarise risk rather than explain it.")

with tab4:
    st.markdown("Each test states its null (H₀) and alternative (H₁) hypothesis. All at α = 0.05, **recomputed live** "
                "on the full cleaned dataset (same methods and hypotheses as the notebook).")
    for i, r in run_tests(len(df), df).iterrows():
        with st.container(border=True):
            st.markdown(f"**Test {i + 1}: {r['Question']}**  ·  _{r['Test']}_")
            hcol, rcol = st.columns([3, 2])
            hcol.markdown(f"**H₀:** {r['H0']}  \n**H₁:** {r['H1']}")
            rcol.markdown(f"**Result:** {r['Result']}  \n**Check:** {r['Robustness check']}  \n**Decision:** {r['Decision']} (α = 0.05)")
    st.info("With ~70K rows tiny differences become significant, so judge effect sizes: V for purpose is weak and Cohen's d for "
            "income is negligible. Test 3 is close to a tautology (grade is set from the rate). Test 4 is partly a grade-mix effect. "
            "Income is uncapped in test 1, as in the notebook (max income is $8.7M), hence the Welch and Mann-Whitney checks. "
            "All default rates come from closed loans only, which overstates defaults for recent and 60-month loans.")

with tab5:
    mfile = MODELS / "metrics.json"
    if not mfile.exists():
        st.info("Run `python train_model.py` to generate model metrics.")
    else:
        m = json.loads(mfile.read_text())
        names = {"grade_only (benchmark)": "Grade only (benchmark)", "borrower_logit": "Borrower-only · logistic",
                 "borrower_gbm": "Borrower-only · gradient boosting", "full_logit": "Full (+grade, rate) · logistic",
                 "full_gbm": "Full (+grade, rate) · gradient boosting"}
        rows = []
        for key, label in names.items():
            r, t = m["random"]["models"][key], m["mature"]["models"][key]
            rows.append({"Model": label, "AUC (random split)": r["roc_auc"], "AUC (out-of-time)": t["roc_auc"],
                         "95% CI (out-of-time)": f"{t['roc_auc_ci95'][0]:.3f}-{t['roc_auc_ci95'][1]:.3f}",
                         "KS (out-of-time)": t["ks"], "Top-decile lift (out-of-time)": t["top_decile_lift"]})
        st.subheader("How well can we rank borrowers by default risk?")
        st.caption("Out-of-time = train on 36-month loans issued 2013-14, test on 36-month loans issued 2015. These are the only "
                   "loans whose outcomes are all known (the data has closed loans only), so this test is not distorted by unfinished loans.")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        R, M = m["random"]["models"], m["mature"]["models"]
        nt = m["diagnostics"]["random_split_borrower_logit_without_term_auc"]
        cal = {c["test_year"]: c for c in m["calibration_by_test_year_36m"]}
        st.markdown(
            f"- **Ranking power is modest** (AUC {min(M['borrower_logit']['roc_auc'], M['grade_only (benchmark)']['roc_auc']):.2f}-"
            f"{max(R['full_logit']['roc_auc'], R['full_gbm']['roc_auc']):.2f}, where 0.5 is a coin flip).\n"
            f"- **Lending Club's grade is a hard benchmark.** Borrower traits alone appear to match it on the random split "
            f"({R['borrower_logit']['roc_auc']:.3f} vs {R['grade_only (benchmark)']['roc_auc']:.3f}), but only thanks to `term`, whose signal is inflated by unfinished loans "
            f"(without it: {nt:.3f}). On the clean out-of-time test grade wins "
            f"({M['grade_only (benchmark)']['roc_auc']:.3f} vs {M['borrower_logit']['roc_auc']:.3f}).\n"
            f"- **Adding grade and rate to borrower traits ranks best** ({M['full_logit']['roc_auc']:.3f} out-of-time), a modest gain over grade alone.\n"
            "- **Gradient boosting adds almost nothing** over logistic regression.\n"
            f"- **Probabilities are well calibrated on finished loans** (predicted vs actual within {cal[2014]['gap_pts']:.1f} and {cal[2015]['gap_pts']:.1f} points for 2014 and 2015). "
            f"The {cal[2016]['gap_pts']:.1f}-point 'under-prediction' on 2016 loans is mostly a data artefact: those loans' outcomes are incomplete, so defaults look too frequent.")
        diag = pd.DataFrame(m["calibration_by_test_year_36m"]).rename(columns={
            "test_year": "Test year", "n_test": "Loans", "actual": "Actual default", "predicted": "Predicted",
            "gap_pts": "Gap (pts)", "auc": "AUC", "outcomes_complete": "Outcomes complete?"})
        st.caption("Calibration check, 36-month loans, trained on earlier years only:")
        st.dataframe(diag, hide_index=True, width="stretch")
        a, b = st.columns(2)
        a.image(str(IMAGES / "model_roc.png"))
        b.image(str(IMAGES / "model_drivers.png"))
        st.image(str(IMAGES / "model_deciles_calibration.png"))

    st.subheader("Approval-policy simulator")
    st.caption("Rank loans by out-of-fold predicted risk (5-fold, so no loan is scored by a model that saw it) "
               "and approve only the safest share. Uses the sidebar filters. Ignores pricing and interest income. Default rates are for closed "
               "loans only, so absolute levels are overstated for recent and 60-month loans; compare approved vs rejected, not the levels.")
    _, oof = fit_scorer(len(df), df)
    sc = pd.DataFrame({"p": oof.loc[f.index], "d": f["default"]}).sort_values("p")
    pct = st.slider("Approve the safest X% of applicants", 10, 100, 70, 5)
    k = max(1, int(len(sc) * pct / 100))
    appr, rej = sc.iloc[:k], sc.iloc[k:]
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Default rate, approved", f"{appr['d'].mean():.1%}", f"{(appr['d'].mean() - sc['d'].mean()) * 100:+.1f} pts vs all", delta_color="inverse")
    s2.metric("Default rate, rejected", f"{rej['d'].mean():.1%}" if len(rej) else "n/a")
    s3.metric("Defaults avoided", f"{rej['d'].sum() / max(sc['d'].sum(), 1):.0%}")
    s4.metric("Good loans turned away", f"{(1 - rej['d']).sum() / max((1 - sc['d']).sum(), 1):.0%}")
    steps = np.arange(10, 101, 10)
    curve = pd.DataFrame({"Share of applicants approved (%)": steps,
                          "Default rate of approved pool (%)": [sc.iloc[:max(1, int(len(sc) * q / 100))]["d"].mean() * 100 for q in steps]})
    st.line_chart(curve, x="Share of applicants approved (%)", y="Default rate of approved pool (%)")
    st.caption("The tighter the approval cut-off, the lower the default rate of what you approve, "
               "at the cost of turning away good borrowers.")

with tab6:
    st.subheader("Estimate default risk for a hypothetical loan")
    st.caption("Borrower-only logistic regression (no Lending Club grade or rate). A teaching baseline, not a credit decision.")
    model, oof = fit_scorer(len(df), df)
    a, b, c = st.columns(3)
    loan_amnt = a.number_input("Loan amount ($)", 1000, 40000, 12000, 500)
    term = a.radio("Term (months)", [36, 60], horizontal=True)
    purpose = a.selectbox("Purpose", all_purposes, index=all_purposes.index("debt_consolidation") if "debt_consolidation" in all_purposes else 0)
    income = b.number_input("Annual income ($)", 3000, 250000, 65000, 1000)
    fico_in = b.slider("FICO score", 660, 850, 690)
    dti = b.slider("Debt-to-income (%)", 0.0, 60.0, 18.0, 0.5)
    emp = b.slider("Employment length (years)", 0, 10, 5)
    home = c.selectbox("Home ownership", ["RENT", "MORTGAGE", "OWN"])
    verif = c.selectbox("Income verification", sorted(df["verification_status"].unique()))
    revol_util = c.slider("Revolving utilisation (%)", 0.0, 100.0, 50.0, 1.0)
    revol_bal = c.number_input("Revolving balance ($)", 0, 200000, 11000, 500)
    d1, d2, d3, d4 = st.columns(4)
    open_acc = d1.number_input("Open accounts", 1, 60, 11)
    total_acc = d2.number_input("Total accounts", 2, 100, 24)
    delinq = d3.number_input("Delinquencies (2 yrs)", 0, 20, 0)
    pub_rec = d4.number_input("Public records", 0, 20, 0)

    row = pd.DataFrame([{
        "loan_amnt": loan_amnt, "term": term, "installment": 0.0, "annual_inc_capped": income,
        "dti": dti, "delinq_2yrs": delinq, "open_acc": open_acc, "pub_rec": pub_rec, "revol_bal": revol_bal,
        "revol_util": revol_util, "total_acc": total_acc, "emp_years": emp, "emp_missing": 0,
        "fico_avg": fico_in, "home_ownership": home, "verification_status": verif, "purpose": purpose,
    }])
    p = float(model.predict_proba(add_features(row)[feature_columns("borrower")])[0, 1])
    pctile = float((oof < p).mean())
    r1, r2, r3 = st.columns(3)
    r1.metric("Predicted default probability", f"{p:.1%}", f"{(p - df['default'].mean()) * 100:+.1f} pts vs portfolio", delta_color="inverse")
    r2.metric("Riskier than", f"{pctile:.0%} of loans")
    r3.metric("Risk decile (1 = safest)", f"{min(10, int(pctile * 10) + 1)}")
    st.progress(min(max(p / 0.6, 0.0), 1.0), text="Scale: 0% to 60% default probability")
    st.warning("The model learns from closed loans only, which overstates defaults for recent and 60-month loans, and on its own it ranks "
               "risk less well than Lending Club's grade. Use it to **rank** applicants, not as a calibrated probability or a credit decision.")

with tab7:
    st.markdown("""
1. **Grades E-G** default at 39-51%: tighten approval or pricing.
2. **60-month terms** default more even within a grade (about 1.1-1.8×, around 1.4× for grades C-E; grade-adjusted odds ratio about 1.6). Limit or price them up, especially for grades C-E. The raw ~2× gap overstates it because 60-month loans skew to weaker grades.
3. **Small-business loans** (29% default, 678 loans): require financials or collateral.
4. **Combine income, DTI and FICO**: verification alone does not protect against default. Income on its own is a weak signal (Cohen's d about 0.1), so don't use it as a sole cut-off.
5. **Use grade plus borrower traits.** Lending Club's grade is a strong baseline; adding borrower traits and the rate gives the best ranking (AUC about 0.68 out-of-time). Borrower traits alone rank slightly worse than the grade.
6. **Measure default on finished loans.** This dataset has closed loans only, so recent and 60-month default rates are overstated. Re-estimate on cohorts old enough to have finished before setting prices.
""")
