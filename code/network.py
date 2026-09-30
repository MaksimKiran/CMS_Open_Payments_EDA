import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

RUN_DIAGNOSTICS = True

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_colwidth", 60)

TOP_N = 15
MIN_RECIPIENTS = 5

# Shorten company names if necessary
def short(s, n=25):
    return s if len(s) <= n else s[: n - 1] + "…"

# Style nodes for visualization
def style(node_list):
    biggest = max(strength[n] for n in node_list)
    return ([colors(community_of[n] % 20) for n in node_list],
            [30 + 800 * strength[n] / biggest for n in node_list])

PROJECT_DIR = r"C:\Users\Maksi\PycharmProjects\DATA_MINING_CMS_PROJECT"
DATA_DIR = os.path.join(PROJECT_DIR, "data")
VIS_DIR = os.path.join(PROJECT_DIR, "visualizations")
OUT_DIR = os.path.join(DATA_DIR, "network")
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

COMPANY = "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name"
SPECIALTY = "Covered_Recipient_Specialty_1"

cs = pd.read_csv(os.path.join(DATA_DIR, "company_specialty.csv"), keep_default_na=False)
cs = cs[cs["total_amount"] > 0].copy()  # we are interested in positive payment relationships

# Amounts are extremely right-skewed, so the edge weight is log(1 + amount), not raw dollars
cs["log_weight"] = np.log1p(cs["total_amount"])
cs["payments_per_recipient"] = cs["n_payments"] / cs["n_recipients"]

# Bipartite Graph
G = nx.Graph()
company_nodes = list(cs[COMPANY].unique())
specialty_nodes = list(cs[SPECIALTY].unique())
G.add_nodes_from(company_nodes, bipartite=0, kind="company")
G.add_nodes_from(specialty_nodes, bipartite=1, kind="specialty")

for _, row in cs.iterrows():
    G.add_edge(row[COMPANY], row[SPECIALTY],
               total_amount=row["total_amount"],
               n_payments=row["n_payments"],
               n_recipients=row["n_recipients"],
               log_weight=row["log_weight"])

# assert nx.is_bipartite(G)

print(f"Nodes: {G.number_of_nodes():,} ({len(company_nodes):,} companies, {len(specialty_nodes):,} specialties)")
print(f"Edges: {G.number_of_edges():,}")

# Degree, weighted degree, centrality
nodes = pd.DataFrame({"node": list(G.nodes)})
nodes["kind"] = nodes["node"].map(lambda n: G.nodes[n]["kind"])

# degree: company -> number of specialties it pays, specialty -> number of companies paying it
nodes["degree"] = nodes["node"].map(dict(G.degree()))

# weighted degree: sum of log_weight over all edges of the node (total log-transformed payment strength)
nodes["weighted_degree"] = nodes["node"].map(dict(G.degree(weight="log_weight")))

# degree centrality for bipartite graphs: degree divided by the size of the opposite node set
nodes["degree_centrality"] = nodes["node"].map(nx.bipartite.degree_centrality(G, company_nodes))

# Communities Louvain
communities = nx.community.louvain_communities(G, weight="log_weight", resolution=1.0, seed=42)
community_of = {}
for cid, members in enumerate(communities):
    for i in members:
        community_of[i] = cid
nodes["community"] = nodes["node"].map(community_of)
modularity = nx.community.modularity(G, communities, weight="log_weight")
print(f"Louvain found {len(communities)} communities (modularity {modularity:.3f})")

companies = nodes[nodes["kind"] == "company"]
specialties = nodes[nodes["kind"] == "specialty"]

print(f"\nCompany degree: median {companies['degree'].median():.0f}, max {companies['degree'].max()}, "
      f"{(companies['degree'] == 1).mean():.1%} pay only one specialty")
print(f"Specialty degree: median {specialties['degree'].median():.0f}, max {specialties['degree'].max()}")

cols = ["node", "degree", "weighted_degree", "degree_centrality", "community"]
print(f"\nTop {TOP_N} companies by degree:\n", companies.nlargest(TOP_N, "degree")[cols].to_string(index=False))
print(f"\nTop {TOP_N} companies by weighted degree:\n",
      companies.nlargest(TOP_N, "weighted_degree")[cols].to_string(index=False))
print(f"\nTop {TOP_N} specialties by degree:\n", specialties.nlargest(TOP_N, "degree")[cols].to_string(index=False))
print(f"\nTop {TOP_N} specialties by weighted degree:\n",
      specialties.nlargest(TOP_N, "weighted_degree")[cols].to_string(index=False))

# Summary
rows = []
for cid, group in nodes.groupby("community"):
    comp = group[group["kind"] == "company"]
    spec = group[group["kind"] == "specialty"]
    rows.append({
        "community": cid,
        "n_companies": len(comp),
        "n_specialties": len(spec),
        "top_companies": "; ".join(short(x, 30) for x in comp.nlargest(5, "weighted_degree")["node"]),
        "top_specialties": "; ".join(spec.nlargest(5, "weighted_degree")["node"]),
    })
community_summary = pd.DataFrame(rows).sort_values("n_specialties", ascending=False)
print("\nCommunity summary:\n", community_summary.to_string(index=False))

# Repeated links
edge_cols = [
    COMPANY,
    SPECIALTY,
    "total_amount",
    "n_payments",
    "n_recipients",
    "payments_per_recipient"
]

print(f"\nTop {TOP_N} company-specialty edges by number of payments:\n",
      cs.nlargest(TOP_N, "n_payments")[edge_cols].to_string(index=False))

repeated = cs[cs["n_recipients"] >= MIN_RECIPIENTS]
print(f"\nTop {TOP_N} edges by payments per recipient (at least {MIN_RECIPIENTS} recipients):\n",
      repeated.nlargest(TOP_N, "payments_per_recipient")[edge_cols].to_string(index=False))

nodes[["node", "kind", "degree", "weighted_degree", "degree_centrality", "community"]] \
    .to_csv(os.path.join(OUT_DIR, "nodes.csv"), index=False)
cs[edge_cols].rename(columns={COMPANY: "company", SPECIALTY: "specialty"}) \
    .to_csv(os.path.join(OUT_DIR, "edges.csv"), index=False)
community_summary.to_csv(os.path.join(OUT_DIR, "communities.csv"), index=False)

# Top companies by weighted degree
top = companies.nlargest(TOP_N, "weighted_degree")
fig, ax = plt.subplots(figsize=(9, 7))
ax.barh([short(x, 40) for x in top["node"]], top["weighted_degree"])
ax.invert_yaxis()
ax.set_xlabel("Weighted degree (sum of log(1 + amount))")
ax.set_title(f"Top {TOP_N} companies by weighted degree")
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "network_top_companies.png"), dpi=150)
plt.close()

# Top 40 companies and top 25 specialties by weighted degree
core_c = companies.nlargest(40, "weighted_degree")["node"].tolist()
core_s = specialties.nlargest(25, "weighted_degree")["node"].tolist()
H = G.subgraph(core_c + core_s)
pos = nx.spring_layout(H, weight="log_weight", seed=42, k=0.6)

strength = dict(zip(nodes["node"], nodes["weighted_degree"]))
colors = plt.get_cmap("tab20")

fig, ax = plt.subplots(figsize=(16, 12))
nx.draw_networkx_edges(H, pos, ax=ax, alpha=0.15,
                       width=[0.1 * d["log_weight"] for _, _, d in H.edges(data=True)])
for node_list, shape in [(core_c, "o"), (core_s, "s")]:
    c, s = style(node_list)
    nx.draw_networkx_nodes(H, pos, nodelist=node_list, node_color=c, node_size=s,
                           node_shape=shape, edgecolors="k", linewidths=0.4, ax=ax)
labels = {n: short(n) for n in core_s + core_c[:10]}
nx.draw_networkx_labels(H, pos, labels=labels, font_size=7, ax=ax)
ax.set_title("Core network: companies (circles), specialties (squares); color = Louvain community")
ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "network_core_graph.png"), dpi=150)
plt.close()

print("\nSaved tables to", OUT_DIR, "and figures to", VIS_DIR)

# Chosen from helper script, good ration between total amount and payments
FOCUS_SPECIALTY = "Independent Medical Examiner"
PHYSICIAN = "Covered_Recipient_Profile_ID"

df = pd.read_parquet(DATA_DIR + r"\general_payments_2024_clean.parquet")
focus = df[df[SPECIALTY] == FOCUS_SPECIALTY]

cp = (
    focus.groupby([COMPANY, PHYSICIAN], observed=True)
    .agg(total_amount=("Total_Amount_of_Payment_USDollars", "sum"),
         n_payments=("Number_of_Payments_Included_in_Total_Amount", "sum"))
    .reset_index()
)
cp["log_weight"] = np.log1p(cp["total_amount"])

G2 = nx.Graph()
company_nodes2 = list(cp[COMPANY].unique())
physician_nodes = list(cp[PHYSICIAN].unique())
G2.add_nodes_from(company_nodes2, bipartite=0, kind="company")
G2.add_nodes_from(physician_nodes, bipartite=1, kind="physician")

for _, row in cp.iterrows():
    G2.add_edge(row[COMPANY], row[PHYSICIAN],
                total_amount = row["total_amount"],
                n_payments = row["n_payments"],
                log_weight=row["log_weight"])


print(f"Number of nodes: {G2.number_of_nodes()}, {len(company_nodes2)} companies and {len(physician_nodes)} physicians.")
print(f"Number of edges: {G2.number_of_edges()}")

nodes2 = pd.DataFrame({"node" : list(G2.nodes)})
nodes2["kind"] = nodes2["node"].map(lambda n: G2.nodes[n]["kind"])
nodes2["degree"] = nodes2["node"].map(dict(G2.degree()))
nodes2["weighted_degree"] = nodes2["node"].map(dict(G2.degree(weight="log_weight")))
nodes2["degree_centrality"] = nodes2["node"].map(nx.bipartite.degree_centrality(G2, company_nodes2))

communities2 = nx.community.louvain_communities(G2, weight="log_weight", seed=42)
community_of2 = {}
for cid, members in enumerate(communities2):
    for i in members:
        community_of2[i] = cid
nodes2["community"] = nodes2["node"].map(community_of2)

print(f"{FOCUS_SPECIALTY}: Louvain found {len(communities2)} communities")
print(nodes2.sort_values("weighted_degree", ascending=False).to_string(index=False))

company_order = nodes2[nodes2["kind"] == "company"].sort_values("weighted_degree", ascending=False)["node"].tolist()
physician_order = nodes2[nodes2["kind"] == "physician"].sort_values("weighted_degree", ascending=False)["node"].tolist()

pos2 = nx.bipartite_layout(G2, company_order, align="vertical")
colors2 = plt.get_cmap("tab20")

fig, ax = plt.subplots(figsize=(10, 14))
nx.draw_networkx_edges(G2, pos2, ax=ax, alpha=0.3)

for node_list, shape in [(company_order, "o"), (physician_order, "s")]:
    c = [colors2(community_of2[n] % 20) for n in node_list]
    nx.draw_networkx_nodes(G2, pos2, nodelist=node_list, node_color=c, node_shape=shape,
                            edgecolors="k", linewidths=0.4, node_size=300, ax=ax)

# Only label companies, physician IDs are meaningless numbers and would just clutter this
company_labels = {n: n for n in company_order}
nx.draw_networkx_labels(G2, pos2, labels=company_labels, font_size=8, ax=ax,
                         horizontalalignment="right")

ax.set_title(f"Company-Physician network: {FOCUS_SPECIALTY} (circles=companies, squares=physicians)")
ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "network_independent_medical_examiner.png"), dpi=150)
plt.close()
plt.close()

nodes2[["node", "kind", "degree", "weighted_degree", "degree_centrality", "community"]] \
    .to_csv(os.path.join(OUT_DIR, "independent_medical_examiner_nodes.csv"), index=False)
