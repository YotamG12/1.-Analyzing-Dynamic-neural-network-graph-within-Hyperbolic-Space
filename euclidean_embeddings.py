import node2vec 
import networkx  as nx
import numpy 
import pandas 
import os
import sys

from utilis import data_utilis




data_path = os.path.abspath("./data/Cora/raw")
adj, features, labels, idx_train, idx_val, idx_test = data_utilis.load_citation_data(
    "cora", use_feats=True, data_path=data_path
)

G=nx.from_scipy_sparse_array(adj)

# Precompute probabilities and generate walks
nodetovec = node2vec.Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)

# Embed nodes
model = nodetovec.fit(window=10, min_count=1, batch_words=4)

# Retrieve the embeddings for all nodes
embeddings = {node: model.wv[node] for node in G.nodes()}

# Example: Print the embedding for node 0
print("Embedding for Node 0:", embeddings[0])
