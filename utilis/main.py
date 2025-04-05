from data_utilis import load_citation_data
import os

data_path = os.path.abspath("./data/Cora/raw")
adj, features, labels, idx_train, idx_val, idx_test = load_citation_data(
    "cora", use_feats=True, data_path=data_path
)


# Get number of nodes
num_nodes = features.shape[0]

# Get feature dimensions
num_features = features.shape[1]

print(f"Number of nodes: {num_nodes}")
print(f"Feature dimensions: {num_features}")
