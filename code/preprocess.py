import pandas as pd
import numpy as np

DATA_DIR = r"C:\Users\Maksi\PycharmProjects\DATA_MINING_CMS_PROJECT\data"
RAW_PATH = DATA_DIR + r"\raw_data.csv"

USECOLS = [
    "Record_ID",
    "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name",
    "Covered_Recipient_Profile_ID",
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

df = pd.read_csv(
    RAW_PATH,
    usecols=USECOLS,
    dtype=DTYPES,
    parse_dates=["Date_of_Payment"],
    low_memory=False,
)
print(f"Loaded {len(df):,} rows")

# Missing values
print(df.isna().sum())

# If company name or amount is missing there's not much we can analyze
df = df.dropna(subset=[
    "Total_Amount_of_Payment_USDollars",
    "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name",
])
df["Covered_Recipient_Specialty_1"] = df["Covered_Recipient_Specialty_1"].fillna("Unknown")

# Duplicates

before = len(df)
df = df.drop_duplicates()
print(f"Dropped {before - len(df):,} duplicate rows")

# Log-transform
df["Log_Amount"] = np.log1p(df["Total_Amount_of_Payment_USDollars"])

# Aggregated tables
company_specialty = (
    df.groupby(["Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name",
                "Covered_Recipient_Specialty_1"], observed=True)
      .agg(total_amount=("Total_Amount_of_Payment_USDollars", "sum"),
           n_payments=("Total_Amount_of_Payment_USDollars", "count"),
           mean_amount=("Total_Amount_of_Payment_USDollars", "mean"))
      .reset_index()
)

company_physician = (
    df.groupby(["Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name",
                "Covered_Recipient_Profile_ID"], observed=True)
      .agg(total_amount=("Total_Amount_of_Payment_USDollars", "sum"),
           n_payments=("Total_Amount_of_Payment_USDollars", "count"))
      .reset_index()
)

# This is really just a specialty aggregated table, since year is constant in this dataset (2024).
specialty_year = (
    df.groupby("Covered_Recipient_Specialty_1", observed=True)
      .agg(total_amount=("Total_Amount_of_Payment_USDollars", "sum"),
           n_payments=("Total_Amount_of_Payment_USDollars", "count"))
      .reset_index()
)

print(f"company_specialty: {len(company_specialty):,} rows")
print(f"company_physician: {len(company_physician):,} rows")
print(f"specialty_year: {len(specialty_year):,} rows")

# Parquet keeps dtypes instead of python re-infering them every read. Useful on this scale
df.to_parquet(DATA_DIR + r"\general_payments_2024_clean.parquet", index=False)
company_specialty.to_csv(DATA_DIR + r"\company_specialty.csv", index=False)
company_physician.to_csv(DATA_DIR + r"\company_physician.csv", index=False)
specialty_year.to_csv(DATA_DIR + r"\specialty_year.csv", index=False)