# =======================================================
# 1. Network Neighbor Check
# =======================================================
import pandas as pd
import numpy as np

# 1-1. ------------ setting ------------
ADJ_PATH = "results/adjacency_matrix.csv"
GO_PATH = "results/go_matrix.csv"
GO_TERM = "GO:1990904"
TOP_N_NEIGHBORS = 10  # top 10 neighbors (conservatively)

CANDIDATES_PATH = "candidates.txt"
with open(CANDIDATES_PATH, 'r') as f:
    CANDIDATES = [line.strip() for line in f if line.strip()]

# 1-2. ---------- Load data ------------
print("Loading data...")
adj = pd.read_csv(ADJ_PATH, index_col=0)
go = pd.read_csv(GO_PATH, index_col=0)

# Attach 4932. prefix to protein ID (to match STRING format in adjacency_matrix.csv)
candidates_full = ['4932.' + p for p in CANDIDATES]

# 1-3. --------- Analyze neighbor network ------------
results = []
for pid in candidates_full:
    short_id = pid.replace('4932.', '')

    if pid not in adj.index:
        print(f"{short_id}: not found in adjacency matrix")
        continue

    # Extract top N neighbors (excluding self)
    neighbors = adj.loc[pid].drop(pid, errors='ignore')
    top_neighbors = neighbors.nlargest(TOP_N_NEIGHBORS).index.tolist()

    # Check if GO_TERM exists in go_matrix
    if GO_TERM not in go.columns:
        print(f"GO term {GO_TERM} not found in go_matrix")
        break

    # Calculate ratio of neighbors that have GO:1990904
    neighbor_go = go.loc[go.index.isin(top_neighbors), GO_TERM]
    n_positive = int(neighbor_go.sum())
    ratio = n_positive / len(top_neighbors) if top_neighbors else 0

    results.append({
        'protein': short_id,
        'top_neighbors': TOP_N_NEIGHBORS,
        f'neighbors_with_{GO_TERM}': n_positive,
        'ratio': round(ratio, 3),
        'confidence': 'HIGH' if ratio >= 0.5 else ('MED' if ratio >= 0.3 else 'LOW')
    })

# 1-4. ------------ Save result ------------
network_df = pd.DataFrame(results).sort_values('ratio', ascending=False)
network_df.to_csv("results/network_validation.csv", index=False)

print(f"\nComplete! Saved to results/network_validation.csv")
print(f"\nConfidence distribution:")
print(network_df['confidence'].value_counts())
print(f"\nTop 10 candidates:")
print(network_df.head(10).to_string(index=False))


# ============================================================
# 2. GO Term Consistency Check
# ============================================================

# 2-1. ----- File path -----------
NOVEL_PATH = "results/novel_predictions.csv"

# GO terms biologically related to this go_term (ribosome assembly) ------ Adjusted whenever changing function
RELATED_TERMS = {
    'GO:0034660': 'ncRNA metabolic process',
    'GO:0006396': 'RNA processing',
    'GO:0051276': 'chromosome organization',
    'GO:0003723': 'RNA binding',
    'GO:0031981': 'nuclear lumen',
    'GO:0005622': 'intracellular',
}

# 2-2. --------- Load the data -----------
novel = pd.read_csv(NOVEL_PATH)

# Extract GO:1990904 candidate proteins
ribo_candidates = novel[novel['go_term'] == GO_TERM]['protein'].tolist()
print(f"{GO_TERM} candidates: {len(ribo_candidates)}")

# 2-3. ----- Check GO term consistency for each candidate protein ---------
rows = []
for protein in ribo_candidates:
    predicted_terms = novel[novel['protein'] == protein]['go_term'].tolist()
    row = {'protein': protein.replace('4932.', '')}
    n_related = 0
    for term, name in RELATED_TERMS.items():
        has = term in predicted_terms
        row[term] = '✓' if has else ''
        if has:
            n_related += 1
    row['n_related_terms'] = n_related
    rows.append(row)

# 2-4. --------- Save the result ----------
go_cons_df = pd.DataFrame(rows).sort_values('n_related_terms', ascending=False)
go_cons_df.to_csv("results/go_consistency.csv", index=False)

print(f"Saved to results/go_consistency.csv")
print()
print(go_cons_df.head(20).to_string(index=False))


# ============================================================
# 3. Final Candidates
# ============================================================

# 3-1. ----- Merge network validation and GO consistency -----------
merged = pd.merge(
    network_df[['protein', 'ratio', 'confidence']],
    go_cons_df[['protein', 'n_related_terms']],
    on='protein',
    how='inner'
)

# 3-2. ----- Calculate final score ------
# Network neighbor ratio + GO consistency (weighted)
merged['final_score'] = merged['ratio'] + merged['n_related_terms'] * 0.1
merged = merged.sort_values('final_score', ascending=False)

# Filter HIGH confidence only
final = merged[(merged['confidence'] == 'HIGH') & (merged['n_related_terms'] >= 2)].copy()

final.to_csv("results/final_candidates.csv", index=False)

print(f"\nFinal Candidates: {len(final)}")
print()
print(final.to_string(index=False))