# 🏦 Lending Club Loan Default Risk Analysis

**Which borrower and loan traits predict default, and how should a lender adjust approval and pricing?**
EDA + hypothesis testing on **69,981 closed Lending Club loans (2013–2016)** using Python, pandas, SciPy and statsmodels.

🔗 **Live dashboard:** _add your Streamlit link here_  ·  📓 Notebooks in [`notebooks/`](notebooks)

## At a glance

| Loans analysed | Default rate | Columns | Period |
|---|---|---|---|
| 69,981 | 20.2% (Charged Off) | 26 (after cleaning) | 2013–2016 |

## Key findings

- **Grade is the strongest signal.** Default climbs from **6.3% (A) to 50.7% (G)**.
- **60-month loans default ~2× more than 36-month** (33.9% vs 15.9%), and the gap shows in every grade.
- **Purpose matters, but weakly.** Small business is the riskiest large category (29.4%); wedding the safest (6.1%). Cramér's V is only 0.057.
- **Lower income and FICO go with default.** Income: $77.4K (paid) vs $70.6K (defaulted). FICO ↔ default r = −0.13; interest rate ↔ default r = 0.27.
- **Verification is not protective.** Verified borrowers default more (23.4%) than unverified (14.5%), likely because risky applicants get verified.

![Default rate by grade](images/default_by_grade.png)
![Default rate by grade and term](images/default_by_grade_term.png)

## Hypothesis tests (α = 0.05)

| # | Question | Test | Result |
|---|---|---|---|
| 1 | Do defaulters earn less? | Two-sample t-test | t = 9.84, p ≈ 8e-23 → reject H₀ |
| 2 | Does default depend on purpose? | Chi-square + Cramér's V | χ² = 226.9, dof = 12, V = 0.057 → reject H₀ |
| 3 | Do interest rates differ by grade? | One-way ANOVA + Tukey | F = 130,166; every grade pair differs → reject H₀ |
| 4 | Do 60-month loans default more? | Two-proportion z-test | z = 50.7, p ≈ 0 → reject H₀ |

> With ~70K rows, tiny differences become "significant". Read effect sizes (V, rate gaps), not p-values alone.

## Recommendations

1. Apply stricter approval or pricing to **grades E–G** (39–51% default).
2. **Limit or price-up 60-month terms** for grades C–G.
3. Add extra checks (financials, collateral) for **small-business** loans.
4. Use **income, DTI and FICO** together; verification alone is not a safeguard.

## Method in brief

Chunked load of the 2.2M-row Kaggle file → keep Fully Paid / Charged Off, issued 2013–2016 → random sample of 70,000 (`random_state=42`) → clean (target flag, numeric term / emp_years, FICO average, income capped at 99th percentile, DTI 0–60) → EDA → 4 hypothesis tests. Leakage columns (e.g. payments, recoveries) were never loaded.

## Run it

```bash
git clone https://github.com/<your-username>/lending-club-default-risk.git
cd lending-club-default-risk
pip install -r requirements.txt
streamlit run app.py
```

Re-running the notebooks needs `data/Raw/accepted_2007_to_2018Q4.csv` from [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club) plus `pip install jupyter seaborn scipy statsmodels matplotlib`. Run `01_Load_and_Sample` then `02_EDA_and_Hypothesis_Tests`.

## Structure

```
app.py                  Streamlit dashboard
notebooks/              01 load, sample, clean · 02 EDA + hypothesis tests
data/processed/         loans_clean.csv (analysis-ready)
images/                 charts used in this README
```

## Limitations

Associations, not causation. Only closed loans are included, so recent 60-month loans still running are missing (survivorship bias). Grade and interest rate are Lending Club's own risk outputs, so they summarise risk rather than explain it. No predictive model was built.

_Data: Lending Club via Kaggle. Educational project (Data Analytics mini project)._
