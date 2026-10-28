import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

pd.set_option("display.width", 200)

TOP_N = 10
Z_CUTOFF = 3            # indicators 1 and 3: "far from the others"
MIN_RECIPIENTS = 30     # indicators 1 and 4: smallest specialty (recipients) we trust
MIN_COMPANIES = 20      # indicator 3: smallest specialty (paying companies) we trust
TOP_PCT_PAIRS = 0.001   # indicator 2: top 0.1% of company-recipient pairs
TOP_PCT_GINI = 0.10     # indicator 4: the 10% of specialties with the highest Gini
DOMINANCE_SHARE = 0.50  # indicator 5: one company pays at least half of a specialty
MIN_PAYERS = 5          # indicator 5: smallest number of paying companies

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
ANOMALY_VIS_DIR = PROJECT_DIR / "visualizations" / "anomaly"
OUT_DIR = DATA_DIR / "anomaly"
for d in (ANOMALY_VIS_DIR, OUT_DIR):
    os.makedirs(d, exist_ok=True)

COMPANY = "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name"
SPECIALTY = "Covered_Recipient_Specialty_1"
PHYSICIAN = "Covered_Recipient_Profile_ID"
AMOUNT = "Total_Amount_of_Payment_USDollars"

# Helper 1 (indicators 1 and 3): z-score of a value inside its own group, small groups are skipped
def add_group_zscore(data, group_col, value_col, min_size):
    out = data[data.groupby(group_col)[value_col].transform("size") >= min_size].copy()
    g = out.groupby(group_col)[value_col]
    out["z_score"] = (out[value_col] - g.transform("mean")) / g.transform("std")
    return out.dropna(subset=["z_score"])

# Helper 2 (all indicators): store flagged rows in a list, print how many were flagged and the top 10
FLAG_FRAMES = []

def flag_and_print(name, flagged, entity_cols, value_col, rule, show_cols, n_eligible):
    flagged = flagged.sort_values(value_col, ascending=False)
    print(f"\n=== {name} ===\nRule: {rule}")
    print(f"Possible red flags: {len(flagged):,} of {n_eligible:,} eligible ({len(flagged) / n_eligible:.2%})")
    print(f"Top {TOP_N}:\n", flagged.head(TOP_N)[show_cols].to_string(index=False))
    entity = [" | ".join(r) for r in flagged[entity_cols].astype(str).itertuples(index=False)]
    FLAG_FRAMES.append(pd.DataFrame({"indicator": name, "entity": entity,
                                     "key_value": flagged[value_col].values, "cutoff": rule}))
    return flagged

def gini(v):  # same formula as stats.py
    v = np.sort(np.asarray(v))
    return (2 * np.sum(np.arange(1, len(v) + 1) * v) - (len(v) + 1) * v.sum()) / (len(v) * v.sum())

# Figure helpers: histogram with a cutoff line, horizontal bars (optional cutoff line)
def save_fig(name):
    plt.tight_layout(); plt.savefig(ANOMALY_VIS_DIR / name, dpi=150); plt.close()

def hist_fig(values, cutoff, xlabel, title, name, log_y=False):
    plt.figure(figsize=(9, 6))
    plt.hist(values, bins=100, log=log_y)
    plt.axvline(cutoff, color="k", ls="--", label=f"Cutoff = {cutoff:g}")
    plt.gca().set(xlabel=xlabel, ylabel="Count" + (" (log scale)" if log_y else ""), title=title)
    plt.legend()
    save_fig(name)

def bar_fig(labels, values, xlabel, title, name, cutoff=None, colors="tab:blue"):
    plt.figure(figsize=(10, 7))
    plt.barh(labels, values, color=colors)
    if cutoff is not None:
        plt.axvline(cutoff, color="k", ls="--", label=f"Cutoff = {cutoff:.3g}")
        plt.legend()
    plt.gca().invert_yaxis()
    plt.gca().set(xlabel=xlabel, title=title)
    save_fig(name)

# Recipient total per specialty, calculated in memory. Only positive totals (log and Gini need them)
df = pd.read_parquet(DATA_DIR / "general_payments_2024_clean.parquet", columns=[PHYSICIAN, SPECIALTY, AMOUNT])
rt = df.groupby([SPECIALTY, PHYSICIAN], observed=True)[AMOUNT].sum().reset_index(name="total_amount")
rt = rt[rt["total_amount"] > 0].copy()

cs = pd.read_csv(DATA_DIR / "aggregates" / "company_specialty.csv", keep_default_na=False)
cs = cs[cs["total_amount"] > 0].copy()
cs["log_total_amount"] = np.log1p(cs["total_amount"])
cp = pd.read_csv(DATA_DIR / "aggregates" / "company_physician.csv", keep_default_na=False, dtype={PHYSICIAN: "string"})

# Indicator 1
# Instruction: вкупен износ за примател кој е многу висок во споредба со истата специјалност (z-score во рамки на специјалноста)
rt["log_total_amount"] = np.log1p(rt["total_amount"])
z1 = add_group_zscore(rt, SPECIALTY, "log_total_amount", MIN_RECIPIENTS)
flag_and_print("1. Recipient total far above own specialty", z1[z1["z_score"] > Z_CUTOFF],
               [SPECIALTY, PHYSICIAN], "z_score",
               f"z > {Z_CUTOFF} on log(total amount) within specialty (min {MIN_RECIPIENTS} recipients)",
               [SPECIALTY, PHYSICIAN, "total_amount", "z_score"], len(z1))
hist_fig(z1["z_score"], Z_CUTOFF, "Z-score of log(total amount) within specialty",
         "Indicator 1: recipient z-scores", "anomaly_1_recipient_zscore.png")

# Indicator 2
# Instruction: многу чести плаќања од една компанија кон ист примател
# Payment counts are whole numbers, so many pairs tie at the cutoff: we keep all ties (count can differ from 0.1%)
cutoff2 = cp["n_payments"].quantile(1 - TOP_PCT_PAIRS)
flag_and_print("2. Very frequent payments, one company to one recipient", cp[cp["n_payments"] >= cutoff2],
               [COMPANY, PHYSICIAN], "n_payments",
               f"n_payments >= {cutoff2:.0f} (top {TOP_PCT_PAIRS:.1%} of company-recipient pairs, ties included)",
               [COMPANY, PHYSICIAN, "n_payments", "total_amount"], len(cp))
hist_fig(cp["n_payments"], cutoff2, "Number of payments per company-recipient pair",
         "Indicator 2: payments per company-recipient pair", "anomaly_2_frequent_payments.png", log_y=True)

# Indicator 3
# Instruction: company-specialty парови со невообичаено голем износ
z3 = add_group_zscore(cs, SPECIALTY, "log_total_amount", MIN_COMPANIES)
ind3 = flag_and_print("3. Company-specialty pair with unusually large amount", z3[z3["z_score"] > Z_CUTOFF],
                      [COMPANY, SPECIALTY], "z_score",
                      f"z > {Z_CUTOFF} on log(total amount) within specialty (min {MIN_COMPANIES} companies)",
                      [COMPANY, SPECIALTY, "total_amount", "n_recipients", "z_score"], len(z3))
top = ind3.head(15)
bar_fig(top[COMPANY].str[:25] + " | " + top[SPECIALTY].str[:25], top["z_score"], "Z-score of log(total amount)",
        "Indicator 3: top 15 company-specialty pairs by z-score", "anomaly_3_large_pairs.png")

# Indicator 4
# Instruction: специјалности со висока концентрација
# Calculate the Gini for each specialty, then flag the 10% with the highest Gini
sg = rt.groupby(SPECIALTY)["total_amount"].agg(n_recipients="size", gini=gini,
                                               top1_share=lambda v: v.max() / v.sum()).reset_index()
sg = sg[sg["n_recipients"] >= MIN_RECIPIENTS]
cutoff4 = sg["gini"].quantile(1 - TOP_PCT_GINI)
flag_and_print("4. Specialty with high concentration", sg[sg["gini"] >= cutoff4],
               [SPECIALTY], "gini", f"Gini >= {cutoff4:.3f} (highest {TOP_PCT_GINI:.0%} of specialties)",
               [SPECIALTY, "n_recipients", "gini", "top1_share"], len(sg))
top = sg.nlargest(15, "gini")
bar_fig(top[SPECIALTY].str[:40], top["gini"], "Gini of recipient totals",
        "Indicator 4: top 15 specialties by concentration (red = possible red flag)", "anomaly_4_concentration.png",
        cutoff=cutoff4, colors=np.where(top["gini"] >= cutoff4, "tab:red", "tab:blue"))

# Indicator 5
# Instruction: компании кои доминираат во одредена специјалност
cs["n_payers"] = cs.groupby(SPECIALTY)["total_amount"].transform("size")
cs["share_of_specialty"] = cs["total_amount"] / cs.groupby(SPECIALTY)["total_amount"].transform("sum")
elig5 = cs[cs["n_payers"] >= MIN_PAYERS]
flag_and_print("5. Company dominates a specialty", elig5[elig5["share_of_specialty"] >= DOMINANCE_SHARE],
               [COMPANY, SPECIALTY], "share_of_specialty",
               f"company share of specialty total >= {DOMINANCE_SHARE:.0%} (min {MIN_PAYERS} paying companies)",
               [COMPANY, SPECIALTY, "total_amount", "share_of_specialty", "n_payers"], len(elig5))
top = elig5.nlargest(15, "share_of_specialty")
bar_fig(top[COMPANY].str[:25] + " | " + top[SPECIALTY].str[:25], top["share_of_specialty"],
        "Company share of specialty total", "Indicator 5: top 15 company shares of a specialty",
        "anomaly_5_dominant_companies.png", cutoff=DOMINANCE_SHARE)

# One CSV with all flagged cases
pd.concat(FLAG_FRAMES, ignore_index=True).to_csv(OUT_DIR / "flagged_cases.csv", index=False)