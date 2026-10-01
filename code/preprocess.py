import pandas as pd
import numpy as np
import os
from pathlib import Path

RUN_DIAGNOSTICS = True

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
os.makedirs(DATA_DIR, exist_ok=True)
RAW_PATH = DATA_DIR / "raw_data.csv"
RAW_CACHE_PATH = DATA_DIR / "raw_data_cache.parquet"
AGG_DIR = DATA_DIR / "aggregates"
os.makedirs(AGG_DIR, exist_ok=True)

USECOLS = [
    "Record_ID",
    "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name",
    "Covered_Recipient_Profile_ID",
    "Teaching_Hospital_ID",
    "Covered_Recipient_Type",
    "Recipient_State",
    "Covered_Recipient_Specialty_1",
    "Total_Amount_of_Payment_USDollars",
    "Date_of_Payment",
    "Nature_of_Payment_or_Transfer_of_Value",
    "Form_of_Payment_or_Transfer_of_Value",
    "Number_of_Payments_Included_in_Total_Amount",
    "Related_Product_Indicator",
    "Name_of_Drug_or_Biological_or_Device_or_Medical_Supply_1",
]

DTYPES = {
    "Record_ID" : "string",
    "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name": "string",
    "Covered_Recipient_Profile_ID": "string",
    "Teaching_Hospital_ID" : "string",
    "Covered_Recipient_Type": "category",
    "Recipient_State": "category",
    "Covered_Recipient_Specialty_1": "string",
    "Total_Amount_of_Payment_USDollars": "float64",
    "Nature_of_Payment_or_Transfer_of_Value": "category",
    "Form_of_Payment_or_Transfer_of_Value": "category",
    "Number_of_Payments_Included_in_Total_Amount": "Int32",
    "Related_Product_Indicator": "category",
    "Name_of_Drug_or_Biological_or_Device_or_Medical_Supply_1": "string",
}

if os.path.exists(RAW_CACHE_PATH):
    df = pd.read_parquet(RAW_CACHE_PATH)
    print(f"Loaded {len(df):,} rows from cache")
else:
    df = pd.read_csv(
        RAW_PATH,
        usecols=USECOLS,
        dtype=DTYPES,
        parse_dates=["Date_of_Payment"],
        low_memory=False,
    )
    df.to_parquet(RAW_CACHE_PATH, index=False)
    print(f"Loaded {len(df):,} rows from CSV and cached to parquet")

if RUN_DIAGNOSTICS:
    print(df["Covered_Recipient_Type"].value_counts())
    print(df["Covered_Recipient_Type"].value_counts(normalize=True) * 100)
    print("NaNs:\n", df.isna().sum())

# Missing values
# If company name or amount is missing there's not much we can analyze
df = df.dropna(subset=[
    "Total_Amount_of_Payment_USDollars",
    "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name",
])
# As per the previous report there is a 1:1 match between nans in specialty and amount of teaching hospitals
df["Covered_Recipient_Specialty_1"] = df["Covered_Recipient_Specialty_1"].fillna("Teaching Hospital")

th_mask = df["Covered_Recipient_Type"].str.contains("Teaching")
df.loc[th_mask, "Covered_Recipient_Profile_ID"] = "TH_" + df.loc[th_mask, "Teaching_Hospital_ID"]
df = df.drop(columns=["Teaching_Hospital_ID"])

if RUN_DIAGNOSTICS:
    print("Here! Recipient ID's nan check.")
    print(df["Covered_Recipient_Profile_ID"].isna().sum())

raw_specialty = df["Covered_Recipient_Specialty_1"]
specialty_parts = raw_specialty.str.split("|")
is_physician = df["Covered_Recipient_Type"] == "Covered Recipient Physician"

if RUN_DIAGNOSTICS:
    segment_counts = specialty_parts.str.len()
    print(pd.crosstab(df["Covered_Recipient_Type"], segment_counts))

segment_2 = specialty_parts.str[1]
segment_last = specialty_parts.str[-1]

extracted = np.where(is_physician, segment_2, segment_last)
df["Covered_Recipient_Specialty_1"] = pd.Series(extracted, index=df.index).fillna(raw_specialty)

# Duplicates
before = len(df)
df = df.drop_duplicates()

if RUN_DIAGNOSTICS:
    print(f"Dropped {before - len(df):,} duplicate rows")

# Company name standardization check
if RUN_DIAGNOSTICS:
    company_names = df["Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name"]

    normalized_names = (
        company_names
        .str.upper()
        .str.replace(r"[^A-Z0-9]", "", regex=True)
    )

    name_variants = pd.DataFrame({
        "original": company_names,
        "normalized": normalized_names
    }).drop_duplicates()

    suspicious = (
        name_variants.groupby("normalized")["original"]
        .agg(list)
    )

    suspicious = suspicious[suspicious.str.len() > 1]

    print(f"Potential company-name variants: {len(suspicious):,}")
    for variants in suspicious.head(20):
        print(variants)

# Standardize company names
company_names = df["Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name"]

normalized_names = (
    company_names
    .str.casefold()
    .str.replace(r"[^a-z0-9]", "", regex=True)
)

canonical_names = (
    pd.DataFrame({
        "normalized": normalized_names,
        "original": company_names
    })
    .groupby("normalized")["original"]
    .agg(lambda x: x.mode().iloc[0])
)

df["Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name"] = (
    normalized_names.map(canonical_names)
)

# Log-transform
df["Log_Amount"] = np.log1p(df["Total_Amount_of_Payment_USDollars"])

# Aggregated tables


company_specialty = (
    df.groupby(["Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name",
                "Covered_Recipient_Specialty_1"], observed=True)
      .agg(total_amount=("Total_Amount_of_Payment_USDollars", "sum"),
           n_payments=("Number_of_Payments_Included_in_Total_Amount", "sum"),
           n_recipients=("Covered_Recipient_Profile_ID", "nunique"))
      .reset_index()
)
company_specialty["mean_amount"] = (
    company_specialty["total_amount"] / company_specialty["n_payments"]
)

company_physician = (
    df.groupby(["Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name",
                "Covered_Recipient_Profile_ID"], observed=True)
      .agg(total_amount=("Total_Amount_of_Payment_USDollars", "sum"),
           n_payments=("Number_of_Payments_Included_in_Total_Amount", "sum"))
      .reset_index()
)

# This is really just a specialty aggregated table, since year is constant in this dataset (2024).
specialty_year = (
    df.groupby("Covered_Recipient_Specialty_1", observed=True)
      .agg(total_amount=("Total_Amount_of_Payment_USDollars", "sum"),
           n_payments=("Number_of_Payments_Included_in_Total_Amount", "sum"))
      .reset_index()
)

if RUN_DIAGNOSTICS:
    print(f"company_specialty: {len(company_specialty):,} rows")
    print(f"company_physician: {len(company_physician):,} rows")
    print(f"specialty_year: {len(specialty_year):,} rows")

# Use parquet for cleaned data because it's faster for large files, keeps dtypes
df.to_parquet(DATA_DIR / "general_payments_2024_clean.parquet", index=False)
company_specialty.to_csv(AGG_DIR / "company_specialty.csv", index=False)
company_physician.to_csv(AGG_DIR / "company_physician.csv", index=False)
specialty_year.to_csv(AGG_DIR / "specialty_year.csv", index=False)