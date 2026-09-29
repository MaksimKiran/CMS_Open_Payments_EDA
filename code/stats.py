import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

PROJECT_DIR = r"C:\Users\Maksi\PycharmProjects\DATA_MINING_CMS_PROJECT"
DATA_DIR = PROJECT_DIR + r"\data"
VIS_DIR = PROJECT_DIR + r"\visualizations"
os.makedirs(VIS_DIR, exist_ok=True)

df = pd.read_parquet(DATA_DIR + r"\general_payments_2024_clean.parquet")

# Raw vs Log amount histogram
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].hist(df["Total_Amount_of_Payment_USDollars"], bins=200)
axes[0].set_yscale("log")  # linear y would hide everything past the first bar
axes[0].set_title("Raw Payment Amount")
axes[0].set_xlabel("Amount (USD)")
axes[0].set_ylabel("Count (log scale)")

axes[1].hist(df["Log_Amount"], bins=200, density=True, alpha=0.6, label="Data")
# We overlay a normal curve, and if Log_Amount follows a normal distribution,
# this should hug the histogram closely
mu, sigma = df["Log_Amount"].mean(), df["Log_Amount"].std()
x = np.linspace(df["Log_Amount"].min(), df["Log_Amount"].max(), 500)
axes[1].plot(x, stats.norm.pdf(x, mu, sigma), "r-", lw=2, label="Fitted Normal")
axes[1].set_title("Log(1 + Payment Amount)")
axes[1].set_xlabel("Log Amount")
axes[1].set_ylabel("Density")
axes[1].legend()

plt.tight_layout()
plt.savefig(VIS_DIR + r"\amount_histograms.png", dpi=150)
plt.close()

# Examining heavy-tailed distribution with Q-Q plot
fig, ax = plt.subplots(figsize=(6, 6))
stats.probplot(df["Log_Amount"], dist="norm", plot=ax)
ax.set_title("Q-Q Plot: Log_Amount vs Normal")
plt.tight_layout()
plt.savefig(VIS_DIR + r"\log_amount_qq_plot.png", dpi=150)
plt.close()

# Concentration of payments through Gini, Lorenz curves and topx% share
recipient_totals = (
    df.groupby("Covered_Recipient_Profile_ID", observed=True)["Total_Amount_of_Payment_USDollars"]
      .sum()
      .values
)

def gini_coefficient(values):
    values = np.sort(values)
    n = len(values)
    cum_values = np.cumsum(values)
    return (2 * np.sum(np.arange(1, n + 1) * values) - (n + 1) * cum_values[-1]) / (n * cum_values[-1])

def lorenz_curve(values):
    values = np.sort(values)
    cum_share = np.cumsum(values) / values.sum()
    cum_share = np.insert(cum_share, 0, 0)
    pop_share = np.linspace(0, 1, len(cum_share))
    return pop_share, cum_share

def top_pct_share(values, pct):
    values_sorted = np.sort(values)[::-1]
    n_top = max(1, int(len(values) * pct))
    return values_sorted[:n_top].sum() / values_sorted.sum()

gini = gini_coefficient(recipient_totals)
print(f"Gini coefficient (recipient totals): {gini:.4f}")

for pct in [0.01, 0.05, 0.10]:
    share = top_pct_share(recipient_totals, pct)
    print(f"Top {pct:.0%} of recipients receive {share:.1%} of total payments")

# Drawing
pop_share, cum_share = lorenz_curve(recipient_totals)
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(pop_share, cum_share, label="Lorenz curve")
ax.plot([0, 1], [0, 1], "k--", label="Perfect equality")
ax.set_xlabel("Cumulative share of recipients")
ax.set_ylabel("Cumulative share of total payments")
ax.set_title(f"Lorenz Curve (Gini = {gini:.3f})")
ax.legend()
plt.tight_layout()
plt.savefig(VIS_DIR + r"\lorenz_curve.png", dpi=150)
plt.close()

# Specialties and nature of payment distributions with violin plots and ECDF
top_specialties = df["Covered_Recipient_Specialty_1"].value_counts().head(10).index
subset = df[df["Covered_Recipient_Specialty_1"].isin(top_specialties)]

fig, ax = plt.subplots(figsize=(10, 6))
sns.violinplot(data=subset, x="Covered_Recipient_Specialty_1", y="Log_Amount", ax=ax)
ax.set_title("Log Amount by Specialty (Top 10 by Payment Count)")
ax.set_xlabel("")
ax.set_ylabel("Log Amount")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(VIS_DIR + r"\log_amount_by_specialty.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(10, 6))
sns.violinplot(data=df, x="Nature_of_Payment_or_Transfer_of_Value", y="Log_Amount", ax=ax)
ax.set_title("Log Amount by Nature of Payment")
ax.set_xlabel("")
ax.set_ylabel("Log Amount")
ax.tick_params(axis="x", rotation=90)
plt.tight_layout()
plt.savefig(VIS_DIR + r"\log_amount_by_payment_type.png", dpi=150)
plt.close()


def save_ecdf(data, group_col, plot_filename, legend_filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    for group_val in data[group_col].unique():
        values = np.sort(data.loc[data[group_col] == group_val, "Log_Amount"])
        y = np.arange(1, len(values) + 1) / len(values)
        ax.plot(values, y, label=group_val)
    ax.set_xlabel("Log Amount")
    ax.set_ylabel("Cumulative proportion")
    ax.set_title(f"ECDF of Log Amount by {group_col}")
    handles, labels = ax.get_legend_handles_labels()
    plt.tight_layout()
    plt.savefig(VIS_DIR + f"\\{plot_filename}.png", dpi=150)
    plt.close()

    fig_legend = plt.figure(figsize=(4, len(labels) * 0.3))
    fig_legend.legend(handles, labels, loc="center")
    plt.axis("off")
    plt.savefig(VIS_DIR + f"\\{legend_filename}.png", dpi=150, bbox_inches="tight")
    plt.close()


save_ecdf(df, "Nature_of_Payment_or_Transfer_of_Value",
          "log_amount_ecdf_by_payment_type", "log_amount_ecdf_by_payment_type_legend")
save_ecdf(subset, "Covered_Recipient_Specialty_1",
          "log_amount_ecdf_by_specialty", "log_amount_ecdf_by_specialty_legend")


# Visualizations for specialty table from step 1, most paid companies by total amount AND by number of payments
specialty_totals = pd.read_csv(DATA_DIR + r"\specialty_year.csv")

top_by_amount = specialty_totals.sort_values("total_amount", ascending=False).head(15)
fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(top_by_amount["Covered_Recipient_Specialty_1"], top_by_amount["total_amount"])
ax.invert_yaxis()
ax.set_xlabel("Total Amount (USD)")
ax.set_title("Top 15 Specialties by Total Payment Amount")
plt.tight_layout()
plt.savefig(VIS_DIR + r"\top_specialties_by_total_amount.png", dpi=150)
plt.close()

top_by_count = specialty_totals.sort_values("n_payments", ascending=False).head(15)
fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(top_by_count["Covered_Recipient_Specialty_1"], top_by_count["n_payments"])
ax.invert_yaxis()
ax.set_xlabel("Number of Payments")
ax.set_title("Top 15 Specialties by Payment Count")
plt.tight_layout()
plt.savefig(VIS_DIR + r"\top_specialties_by_payment_count.png", dpi=150)
plt.close()
print("Saved visualizations in folder.")