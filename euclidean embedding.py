from node2vec import Node2Vec
import networkx as nx
import os
from utilis import data_utilis
import scipy.sparse

data_path = os.path.abspath("./1.-Analyzing-Dynamic-neural-network-graph-within-Hyperbolic-Space/data/Cora/raw")
adj, features, labels, idx_train, idx_val, idx_test = data_utilis.load_citation_data(
    "cora", use_feats=True, data_path=data_path
)

# Precompute probabilities and generate walks
node2vec = Node2Vec(adj, dimensions=64, walk_length=30, num_walks=200, workers=4)

# Embed nodes
model = node2vec.fit(window=10, min_count=1, batch_words=4)

# Retrieve the embeddings for all nodes
embeddings = {node: model.wv[node] for node in adj.nodes()}

# Example: Print the embedding for node 0
print("Embedding for Node 0:", embeddings[0])

