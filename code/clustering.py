import os
import textwrap

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUT_DIR = DATA_DIR / "clustering"
VIS_DIR = PROJECT_DIR / "visualizations" / "clustering"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)

COMPANY = "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name"
SPECIALTY = "Covered_Recipient_Specialty_1"
FORM = "Form_of_Payment_or_Transfer_of_Value"

df = pd.read_parquet(DATA_DIR / "general_payments_2024_clean.parquet")

# Build the basic payment profile for each specialty
profile = (
    df.groupby(SPECIALTY, observed=True)
      .agg(
          n_payments=("Number_of_Payments_Included_in_Total_Amount", "sum"),
          total_amount=("Total_Amount_of_Payment_USDollars", "sum"),
          n_companies=(COMPANY, "nunique"),
      )
      .reset_index()
)
profile["mean_amount"] = profile["total_amount"] / profile["n_payments"]

# Log transform because extremely skewed
profile["log_total_amount"] = np.log1p(profile["total_amount"])
profile["log_mean_amount"] = np.log1p(profile["mean_amount"])

# Build a table showing how much each specialty paid through each payment type
form_amounts = (
    df.groupby([SPECIALTY, FORM], observed=True)["Total_Amount_of_Payment_USDollars"]
      .sum()
      .unstack(fill_value=0)
)

form_shares = form_amounts.div(form_amounts.sum(axis=1), axis=0)
form_shares = form_shares.add_prefix("share_")

profile = profile.merge(form_shares, on=SPECIALTY)

# These are the columns we will actually cluster on
feature_cols = ["n_payments", "log_total_amount", "log_mean_amount", "n_companies"]
feature_cols = feature_cols + list(form_shares.columns)

features = profile[feature_cols]

# KMeans needs all features on the same scale, so we standardize them first
# Without this, total_amount (millions) would completely dominate the shares (0 to 1)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Try a few cluster counts and print the silhouette score for each one
# A higher silhouette score means the clusters are more clearly separated
# Also print the smallest cluster size, a high score from a tiny leftover
# cluster of 1 or 2 odd specialties is not a real, useful grouping
print("Choosing number of clusters:")
for k in range(2, 16):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(features_scaled)
    score = silhouette_score(features_scaled, labels)
    smallest_cluster = pd.Series(labels).value_counts().min()
    print(f"k={k}, silhouette score={score:.3f}, smallest cluster size={smallest_cluster}")


inertias = []
k_range = range(2, 16)
for k in k_range:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(features_scaled)
    inertias.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(list(k_range), inertias, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Plot")
plt.tight_layout()
plt.savefig(VIS_DIR / "elbow_plot.png", dpi=150)
plt.close()


K = 5
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
profile["cluster"] = kmeans.fit_predict(features_scaled)

print(f"\nUsing k={K}")
print("Number of specialties per cluster:")
print(profile["cluster"].value_counts().sort_index())

# Save the full profile table with cluster labels for later use
profile.to_csv(OUT_DIR / "specialty_clusters.csv", index=False)

# For the heatmap, add the scaled features back onto the profile table
scaled_df = pd.DataFrame(features_scaled, columns=feature_cols)
scaled_df["cluster"] = profile["cluster"]

# Average scaled feature values per cluster
# This tells us what makes each cluster different from the others
cluster_profile = scaled_df.groupby("cluster").mean()

# Only show the main numeric features and the top payment type shares
# Too many columns would make the heatmap unreadable
top_form_cols = form_shares.mean().sort_values(ascending=False).head(6).index.tolist()
heatmap_cols = ["n_payments", "log_total_amount", "log_mean_amount", "n_companies"] + top_form_cols
heatmap_data = cluster_profile[heatmap_cols]
print("\nCluster profile (standardized feature averages):")
print(heatmap_data.round(2).to_string())

# Standardized values only show statistical outlierness, not real-world size
# Print the raw (unscaled) averages too, so small percentages are not mistaken for big ones
raw_cluster_profile = profile.groupby("cluster")[heatmap_cols].mean()
print("\nCluster profile (raw values, not standardized):")
print(raw_cluster_profile.round(4).to_string())

# Wrap long feature names so the labels fit.
wrapped_labels = [
    textwrap.fill(label.replace("share_", ""), width=18)
    for label in heatmap_cols
]

plt.figure(figsize=(16, 6))
sns.heatmap(heatmap_data,annot=True,fmt=".2f",cmap="coolwarm",center=0)
plt.title("Cluster Profiles (standardized feature averages)")
plt.xlabel("Feature")
plt.ylabel("Cluster")
plt.xticks(
    np.arange(len(wrapped_labels)) + 0.5,wrapped_labels,rotation=0
)
plt.tight_layout()
plt.savefig(VIS_DIR / "cluster_profile_heatmap.png", dpi=150)
plt.close()

# Print which specialties ended up in each cluster, useful for sanity checking
print("\nSpecialties per cluster:")
for cluster_id in sorted(profile["cluster"].unique()):
    names = profile[profile["cluster"] == cluster_id][SPECIALTY].tolist()
    print(f"\nCluster {cluster_id} ({len(names)} specialties):")
    print(names)

print("\nSaved specialty_clusters.csv and cluster_profile_heatmap.png")