"""Lending Club Default Risk Dashboard.  Run: streamlit run app.py"""
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Lending Club Default Risk", page_icon="🏦", layout="wide")

DATA = Path(__file__).parent / "data" / "processed" / "loans_clean.csv"
GRADES = list("ABCDEFG")


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
grades = st.sidebar.multiselect("Grade", GRADES, default=GRADES)
terms = st.sidebar.multiselect("Term (months)", [36, 60], default=[36, 60])
all_purposes = sorted(df["purpose"].unique())
purposes = st.sidebar.multiselect("Purpose", all_purposes, default=all_purposes)
years = st.sidebar.slider("Issue year", int(df["issue_year"].min()), int(df["issue_year"].max()),
                          (int(df["issue_year"].min()), int(df["issue_year"].max())))
fico = st.sidebar.slider("FICO score", int(df["fico_avg"].min()), int(df["fico_avg"].max()),
                         (int(df["fico_avg"].min()), int(df["fico_avg"].max())))

f = apply_filters(df, grades, terms, purposes, years, fico)
if f.empty:
    st.warning("No loans match these filters.")
    st.stop()

# ---------- header + KPIs ----------
st.title("🏦 Lending Club Loan Default Risk")
st.caption("69,981 closed loans, 2013-2016 · default = Charged Off")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Loans", f"{len(f):,}")
c2.metric("Default rate", f"{f['default'].mean():.1%}", f"{(f['default'].mean() - df['default'].mean()) * 100:+.1f} pts vs all")
c3.metric("Avg interest rate", f"{f['int_rate'].mean():.1f}%")
c4.metric("Avg FICO", f"{f['fico_avg'].mean():.0f}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Default drivers", "Grade × Term", "Credit signals", "Hypothesis tests", "Recommendations"])

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
    b.caption("Verified borrowers default more: risky applicants are the ones who get verified.")

with tab2:
    pivot = f.pivot_table(values="default", index="grade", columns="term", aggfunc="mean", observed=True) * 100
    pivot.index = pivot.index.astype(str)
    pivot.columns = [f"{c} months" for c in pivot.columns]
    st.bar_chart(pivot.round(1))
    st.dataframe(pivot.round(1), use_container_width=True)
    st.caption("60-month loans default more in every grade; the gap is widest for grades C-E.")

with tab3:
    a, b = st.columns(2)
    fico_band = pd.cut(f["fico_avg"], bins=[600, 660, 680, 700, 720, 740, 760, 900]).astype(str)
    a.subheader("Default rate (%) by FICO band")
    a.bar_chart(default_rate(f.assign(fico_band=fico_band), "fico_band"))
    rate_band = pd.cut(f["int_rate"], bins=[0, 8, 11, 14, 17, 20, 25, 40]).astype(str)
    b.subheader("Default rate (%) by interest-rate band")
    b.bar_chart(default_rate(f.assign(rate_band=rate_band), "rate_band"))
    st.caption("Grade and interest rate are Lending Club's own risk outputs, so they summarise risk rather than explain it.")

with tab4:
    st.markdown("All tests at α = 0.05 on the full cleaned dataset (static results from the notebook).")
    st.dataframe(pd.DataFrame({
        "Question": ["Do defaulters earn less?", "Does default depend on purpose?",
                     "Do interest rates differ by grade?", "Do 60-month loans default more?"],
        "Test": ["Two-sample t-test", "Chi-square + Cramér's V", "One-way ANOVA + Tukey", "Two-proportion z-test"],
        "Result": ["t = 9.84, p ≈ 8e-23", "χ² = 226.9, dof = 12, V = 0.057",
                   "F = 130,166, all grade pairs differ", "z = 50.7, p ≈ 0 (33.9% vs 15.9%)"],
        "Decision": ["Reject H₀"] * 4,
    }), hide_index=True, use_container_width=True)
    st.info("With ~70K rows tiny differences become significant. Judge by effect size (V = 0.057 is weak), not p-values alone.")

with tab5:
    st.markdown("""
1. **Grades E-G** default at 39-51%: tighten approval or pricing.
2. **60-month terms** default about 2× more: limit them or price them up for grades C-G.
3. **Small-business loans** (29% default): require financials or collateral.
4. **Combine income, DTI and FICO**: verification alone does not protect against default.
""")
