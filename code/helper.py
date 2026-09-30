import pandas as pd
import os

# ------------------------Check main graph size----------------------------------------------------------------
DATA_DIR = r"C:\Users\Maksi\PycharmProjects\DATA_MINING_CMS_PROJECT\data"
company_specialty = pd.read_csv(DATA_DIR + r"\company_specialty.csv")

print(f"Rows (company-specialty edges): {len(company_specialty):,}")
print(f"Distinct companies: {company_specialty['Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name'].nunique():,}")
print(f"Distinct specialties: {company_specialty['Covered_Recipient_Specialty_1'].nunique():,}")

n_companies = company_specialty["Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name"].nunique()
n_specialties = company_specialty["Covered_Recipient_Specialty_1"].nunique()
max_possible_edges = n_companies * n_specialties

print(f"Total nodes (companies + specialties): {n_companies + n_specialties:,}")
print(f"Max possible edges (fully connected bipartite): {max_possible_edges:,}")
print(f"Actual edges / max possible: {len(company_specialty) / max_possible_edges:.4%}")
# ------------------------Check NOPIITA Column is not always 1----------------------------------------------------------------
# DATA_DIR = r"C:\Users\Maksi\PycharmProjects\DATA_MINING_CMS_PROJECT\data"
# df = pd.read_parquet(os.path.join(DATA_DIR, "general_payments_2024_clean.parquet"))
# print(df["Number_of_Payments_Included_in_Total_Amount"].value_counts())

# -----------------------Check graph size for company-physician -------------------------------------------------------------------
# import pandas as pd
#
# DATA_DIR = r"C:\Users\Maksi\PycharmProjects\DATA_MINING_CMS_PROJECT\data"
#
# df = pd.read_parquet(DATA_DIR + r"\general_payments_2024_clean.parquet")
#
# COMPANY = "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name"
# SPECIALTY = "Covered_Recipient_Specialty_1"
# PHYSICIAN = "Covered_Recipient_Profile_ID"
#
# summary = (
#     df.groupby(SPECIALTY, observed=True)
#       .agg(
#           n_companies=(COMPANY, "nunique"),
#           n_physicians=(PHYSICIAN, "nunique"),
#           total_amount=("Total_Amount_of_Payment_USDollars", "sum"),
#           n_payments=("Number_of_Payments_Included_in_Total_Amount", "sum"),
#       )
#       .reset_index()
# )
#
# # Distinct company-physician pairs per specialty, the actual edge count
# edges_per_specialty = (
#     df.drop_duplicates(subset=[SPECIALTY, COMPANY, PHYSICIAN])
#       .groupby(SPECIALTY, observed=True)
#       .size()
#       .rename("n_edges")
# )
# summary = summary.merge(edges_per_specialty, on=SPECIALTY)
#
# summary["n_nodes"] = summary["n_companies"] + summary["n_physicians"]
# summary["mean_payment"] = summary["total_amount"] / summary["n_payments"]
#
# print("Smallest networks (by node count):")
# print(summary.sort_values("n_nodes").head(15).to_string(index=False))
#
# print("\nHighest mean payment for comparison:")
# print(summary.sort_values("mean_payment", ascending=False).head(15).to_string(index=False))