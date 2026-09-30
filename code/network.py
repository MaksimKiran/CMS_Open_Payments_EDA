import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

RUN_DIAGNOSTICS = True

PROJECT_DIR = r"C:\Users\Maksi\PycharmProjects\DATA_MINING_CMS_PROJECT"
DATA_DIR = os.path.join(PROJECT_DIR, "data")
VIS_DIR = os.path.join(PROJECT_DIR, "visualizations")
OUT_DIR = os.path.join(DATA_DIR, "network")
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

COMPANY = "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name"
SPECIALTY = "Covered_Recipient_Specialty_1"

TOP_N = 15
MIN_RECIPIENTS = 5

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_colwidth", 60)


# Shorten company names if necessary
def short(s, n=25):
    return s if len(s) <= n else s[: n - 1] + "…"

# Style nodes for visualization
def style(node_list):
    biggest = max(strength[n] for n in node_list)
    return ([colors(community_of[n] % 20) for n in node_list],
            [30 + 800 * strength[n] / biggest for n in node_list])


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
if RUN_DIAGNOSTICS:
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
community_of = {n: cid for cid, members in enumerate(communities) for n in members}
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
