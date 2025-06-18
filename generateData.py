import os
import ast
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import LabelBinarizer
sys.stdout.reconfigure(encoding='utf-8')
# === CONFIG ===
INPUT_CSV = INPUT_CSV = r"C:\AdiHS Project\data\final_filtered_by_fos_and_reference.csv"

OUTPUT_DIR = "C:\AdiHS Project\data\generate_custom_output"
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
vectorizer = CountVectorizer()  # Use all available features
features = vectorizer.fit_transform(df['abstract'].fillna("")).toarray()


# === LABEL ENCODING ===
lb = LabelBinarizer()
labels = lb.fit_transform(df['fos.name'])

# === GRAPH ===
def parse_refs(refs):
    try:
        return [id2idx[r] for r in ast.literal_eval(refs) if r in id2idx]
    except:
        return []

graph = {row['node_idx']: parse_refs(row['references']) for _, row in df.iterrows()}

# === SPLIT MATRICES ===
x = features[train_idx]
y = labels[train_idx]
tx = features[test_idx]
ty = labels[test_idx]
allx = features[train_idx + val_idx]
ally = labels[train_idx + val_idx]

# === SAVE ===
def save_pickle(data, name):
    with open(os.path.join(OUTPUT_DIR, f"ind.dblpv13.{name}"), "wb") as f:
        pickle.dump(data, f)

save_pickle(x, "x")
save_pickle(y, "y")
save_pickle(tx, "tx")
save_pickle(ty, "ty")
save_pickle(allx, "allx")
save_pickle(ally, "ally")
save_pickle(graph, "graph")

# Save test.index
with open(os.path.join(OUTPUT_DIR, "ind.dblpv13.test.index"), "w") as f:
    for idx in test_idx:
        f.write(f"{idx}\n")

print(f"✅ Files saved to {OUTPUT_DIR}")
