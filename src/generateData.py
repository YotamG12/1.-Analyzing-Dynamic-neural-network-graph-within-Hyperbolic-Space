import os
import ast
import sys
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from config import args

sys.stdout.reconfigure(encoding='utf-8')

# === CONFIG ===
INPUT_CSV = r"src/data/final_filtered_by_fos_and_reference.csv"
OUTPUT_DIR = r"src/data/generate_custom_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD DATA ===
df = pd.read_csv(INPUT_CSV)
paper_ids = df['id'].tolist()
id2idx = {pid: i for i, pid in enumerate(paper_ids)}
df['node_idx'] = df['id'].map(id2idx)

# === Split by paper counts grouped by year ===
df_sorted = df.sort_values(by='year').reset_index(drop=True)

total_papers = len(df_sorted)
train_end = int(total_papers * 0.6)
val_end = int(total_papers * 0.8)

train_idx = df_sorted.iloc[:train_end]['node_idx'].tolist()
val_idx = df_sorted.iloc[train_end:val_end]['node_idx'].tolist()
test_idx = df_sorted.iloc[val_end:]['node_idx'].tolist()

print(f"✅ Split by paper counts")
print(f"Number of training papers   : {len(train_idx)}")
print(f"Number of validation papers : {len(val_idx)}")
print(f"Number of test papers       : {len(test_idx)}")

# === FEATURE MATRIX ===
vectorizer = CountVectorizer()
features = vectorizer.fit_transform(df['abstract'].fillna(""))

# === LABEL ENCODING ===
lb = LabelBinarizer()
labels = lb.fit_transform(df['fos.name'])

# === GRAPH (build adjacency list using references) ===
def parse_refs(refs):
    """
    Parse references from a string and map to node indices.

    Args:
        refs (str): String representation of reference IDs.
    Returns:
        list: List of node indices for valid references.
    """
    try:
        return [id2idx[r] for r in ast.literal_eval(refs) if r in id2idx]
    except:
        return []

adjacency_list = {row['node_idx']: parse_refs(row['references']) for _, row in df.iterrows()}

# === DYNAMIC SNAPSHOT CONSTRUCTION FROM TIME BINS ===
print("📦 Building dynamic snapshots from time bins...")
df = df.dropna(subset=['references', 'year'])
df['year'] = df['year'].astype(int)

T = args.Time_stamps
min_year = df['year'].min()
max_year = df['year'].max()
bins = np.linspace(min_year, max_year + 1, T + 1, dtype=int)
df['time_bin'] = pd.cut(df['year'], bins=bins, labels=False, include_lowest=True)


def parse_refs_str(refs):
    """
    Parse references from a string to a Python list.

    Args:
        refs (str): String representation of reference IDs.
    Returns:
        list: List of reference IDs.
    """
    try:
        return ast.literal_eval(refs) if isinstance(refs, str) else []
    except Exception:
        return []

df['parsed_refs'] = df['references'].apply(parse_refs_str)

snapshots = defaultdict(nx.DiGraph)
id_to_idx = {pid: idx for idx, pid in enumerate(df['id'])}

for _, row in df.iterrows():
    t = row['time_bin']
    paper_id = row['id']
    paper_idx = id_to_idx.get(paper_id)
    if paper_idx is None or pd.isna(t):
        continue
    snapshots[int(t)].add_node(paper_idx)
    for ref_id in row['parsed_refs']:
        ref_idx = id_to_idx.get(ref_id)
        if ref_idx is not None:
            snapshots[int(t)].add_edge(paper_idx, ref_idx)

# === SPLIT MATRICES ===
x = features[train_idx]
y = labels[train_idx]
tx = features[test_idx]
ty = labels[test_idx]
allx = features[train_idx + val_idx]
ally = labels[train_idx + val_idx]

# === SAVE ===
def save_pickle(data, name):
    """
    Save data to a pickle file in the output directory.

    Args:
        data: Data to save.
        name (str): Name for the output file (without extension).
    Returns:
        None. Writes file to disk.
    """
    with open(os.path.join(OUTPUT_DIR, f"ind.dblpv13.{name}"), "wb") as f:
        pickle.dump(data, f)

save_pickle(x, "x")
save_pickle(y, "y")
save_pickle(tx, "tx")
save_pickle(ty, "ty")
save_pickle(allx, "allx")
save_pickle(ally, "ally")
save_pickle(adjacency_list, "graph")
save_pickle(dict(snapshots), "snapshot_graphs")

# Save test index
with open(os.path.join(OUTPUT_DIR, "ind.dblpv13.test.index"), "w") as f:
    for idx in test_idx:
        f.write(f"{idx}\n")

print(f"✅ All files saved to {OUTPUT_DIR}")
