# CMS Open Payments 2024: Company and Specialty Payment Analysis

This project analyzes the CMS Open Payments **General Payments** data for **Program Year 2024**. It looks for concentration, network structure and unusual patterns in payments from companies to medical specialties and recipients.

**Important:** CMS states that these financial links do not necessarily mean anything improper. Results are described only as patterns, concentration and possible red flags, never as proven irregularities.

## How to run

1. Install the packages with `pip install -r requirements.txt`.
2. Create a `data/` folder next to `code/` and put `raw_data_cache.parquet` in it (download link [here](https://drive.google.com/file/d/18exXLypxnPEXEf3tm25w0WIptViwFNHl/view?usp=sharing)).
3. Run the scripts in this order: `preprocess.py`, `stats.py`, `network.py`, `clustering.py`.

All paths are relative to the script location. Tables are saved to `data/` and figures are saved to `visualizations/`.

## Main question

**Do the financial links between companies and medical specialties or physicians show concentration, network structure and unusual patterns?**

This is split into smaller questions. Each one is answered by one script (steps 1 to 4 below), and the five additional questions from the instructions are mapped to those scripts at the end.

## Methodology:

## Step 1: `preprocess.py`

**Question: Is the data clean and comparable enough to analyze?**


The input file is already the General Payments data for 2024, so no extra filtering is needed. Only the needed columns are loaded, and the result is cached as parquet so later runs are fast. Rows without an amount or a company name are dropped, since they cannot be analyzed. Teaching hospitals have no specialty or ID (`Covered_recipient_Profile_ID`), so we artificially list their specialty as "Teaching Hospital" and identify them by their hospital ID with the prefix `TH_`. Exact duplicate rows are then removed, however this dataset proved to have no exact duplicates.

The names of the companies are not standardized in the original dataset. To standardize, company names are lowercased and stripped of punctuation and spaces, and names that match after this are treated as the same company and given their most common spelling.

The raw specialty is a `|`-separated hierarchy which follows the [standardized provider taxonomy code list](https://taxonomy.nucc.org/).
It has 3 levels: Provider Grouping | Classification | Area of Specialization. The desired level of granularity lies usually at segment 2 for physicians and segment 3 for non-physicians practitioners. Each label is searchable in the taxonomy code list. Example:

- Allopathic & Osteopathic Physicians | Pediatrics → Pediatrics (physician) 
- Physician Assistants & Advanced Practice Nursing Providers | Nurse Practitioner | Family → Family (non-physician)


Because amounts are extremely skewed, we add `Log_Amount = log(1 + amount)`. Finally we build three aggregated tables in `data/aggregates/`: `company_specialty`, `company_physician` and `specialty_year`. Since all data is from 2024, `specialty_year` is simply the totals per specialty.

Descriptions for all columns within the dataset can be found in CMS' [data dictionary.](https://openpaymentsdata.cms.gov/dataset/e6b17c6a-2534-4207-a4a1-6746a14911ff#data-dictionary)

## Step 2: `stats.py`

**Question: What does the payment distribution look like, and how concentrated is it?**

To see the shape, we draw histograms of the raw and log amounts. A normal curve is drawn over the log histogram, and a Q-Q plot compares `Log_Amount` to a normal distribution. A close fit means the amounts are roughly lognormal, while deviations in the tails show a heavy tail.

To measure concentration, we compute the total payment per recipient and use it for the **Gini coefficient**, the **Lorenz curve** and the share of money going to the top 1%, 5% and 10% of recipients.

**Question: How does the distribution differ between specialties and payment types?**

We draw violin plots and ECDF curves of `Log_Amount`, once by payment form (all forms) and once by specialty (the top 10 by payment count). Bar charts then show the top 15 specialties by total amount and by number of payments.

## Step 3: `network.py` (main part)

**Question: How are companies and specialties connected, who is central, and are there groups?**

We build a bipartite graph with companies on one side and specialties on the other. An edge means the company paid that specialty, and its weight is `log(1 + total amount)` so that a few giant payments do not dominate.

For every node we compute three measures. *Degree* counts how many partners it has, *weighted degree* is the sum of its edge weights (the money strength), and *degree centrality* is the degree relative to the size of the other group in the graph.
The full company - specialty bipartite graph is quite large with 1,961 nodes and 26,809 edges, so we construct a smaller "core" graph of the top 40 companies and top 25 specialties by weighted degree for visualization (spoiler: it still looks bad).

To find groups, **Louvain** community detection was used  with the edge weights and a fixed seed of 42. It finds sets of companies and specialties that are more connected to each other than to the rest. Modularity shows how clear the grouping is, and a summary table lists the top members of each community.

Finally repeated links are explored. Company-specialty pairs are ranked by number of payments and by payments per recipient, and for the second ranking we keep only pairs with at least 5 recipients to avoid tiny cases. The same analysis is repeated for a company-physician network of one specialty (*Independent Medical Examiner*) as a filtered example.

## Step 4: `clustering.py`

**Question: Do specialties fall into natural groups by payment profile?**

Each specialty gets a profile made of its number of payments, total amount, mean amount, number of paying companies and the share of its money paid in each payment form. Totals and means are log-transformed, and all features are standardized so that large numbers do not dominate the 0 to 1 shares.

To choose the number of clusters, we try K-means for k from 2 to 15 and check the silhouette score, the smallest cluster size (to avoid clusters of only 1 or 2 odd specialties) and an elbow plot. Based on the plot and the scores, we use K = 5.

To interpret the result, a heatmap of standardized averages is generated. The raw averages and the list of specialties in each cluster are also printed, so we can check that the clusters make sense.

## Step 5: `anomaly.py`

**Question: Which recipients, links and specialties look unusual compared with their own group?**

We use five simple indicators instead of a complex model. Each case is compared with its own group (the same specialty), so a big payment in a small specialty is not hidden by a big specialty. The script only uses the tables from step 1 and the clean parquet, and the recipient totals per specialty are calculated in memory. Amounts that are zero or negative are skipped, because the log and the Gini need positive numbers.

| # | Indicator | Rule | Why this cutoff |
|---|---|---|---|
| 1 | Recipient far above own specialty | z-score > 3 on log(total amount), within the specialty (min 30 recipients) | z > 3 is a standard "far from the others" rule. We use logs because dollars are very skewed. |
| 2 | Very frequent payments from one company to one recipient | Top 0.1% of company-recipient pairs by number of payments (≥ 71 payments) | There are 4.6M pairs. The top 1% would be about 46,000 cases, which is too many to use. |
| 3 | Unusually large company-specialty pair | z-score > 3 on log(total amount) among the pairs of the same specialty (min 20 companies) | Same idea as indicator 1, but for pairs. |
| 4 | Concentrated specialty | Gini is calculated for each specialty, then the 10% with the highest Gini are flagged (min 30 recipients) | Inequality is high everywhere (overall Gini 0.94), so a fixed cutoff would flag almost everything. A relative cutoff shows which specialties are *more* concentrated than the rest. |
| 5 | Company dominates a specialty | One company pays at least 50% of the specialty total (min 5 paying companies) | "More than half" is easy to explain. The minimum avoids trivial cases with only one or two payers. |

Minimum group sizes are used because a z-score is not stable in tiny groups.

The script prints how many cases each indicator flags and the top 10 of each. All flagged cases are saved in one file, `data/anomaly/flagged_cases.csv`, and there is one figure per indicator in `visualizations/anomaly/`.

**It is important to note** that a flag only means "unusual compared with the group". It does not mean something improper happened. Some flags are small dollar amounts, and some indicators can point to the same payment (for example, indicators 4 and 5 for the same specialty).
## Additional questions: where each is answered

1. **Which specialties receive the most payments?** \
We rank specialties by total amount and by payment count using bar charts in `stats.py`, which are based on the `specialty_year` table. The network script adds the same ranking by weighted degree.

2. **Do a small number of recipients receive a large share of the total?** \
Gini coefficient, Lorenz curve and the top 1%, 5% and 10% shares, all computed on per-recipient totals in `stats.py`.

3. **Which companies are most central in the network?** \
Companies are ranked by degree, weighted degree and degree centrality in `network.py`. Degree shows breadth (how many specialties a company pays) and weighted degree shows money strength.

4. **Are there groups of connected companies and specialties?** We use the Louvain communities of the bipartite graph in `network.py`. The K-means clustering of specialties in `clustering.py` gives a second, independent view to compare against.

5. **Which company-specialty links look unusual?**
We have the repeated-link tables in `network.py`, which show pairs with many payments or many payments per recipient, as well as the indicators in `anomaly.py`.

