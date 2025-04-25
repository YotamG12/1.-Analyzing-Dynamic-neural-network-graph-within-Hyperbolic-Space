import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import from_scipy_sparse_matrix
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from config import args
from model import AdiHS
from utils.data_utilis import load_citation_data
from node2vec import Node2Vec
import networkx as nx

# === Step 1: Load Cora dataset ===
adj, features, labels, idx_train, idx_val, idx_test = load_citation_data(
    dataset_str='cora', use_feats=True, data_path='./data/Cora/raw'
)
edge_index, _ = from_scipy_sparse_matrix(adj)

features = torch.FloatTensor(features.todense()).to(args.device)
labels = torch.LongTensor(labels).to(args.device)
edge_index = edge_index.to(args.device)

# === Step 2: Node2Vec Embedding ===
G = nx.from_scipy_sparse_matrix(adj)
node2vec = Node2Vec(G, dimensions=args.nfeat, walk_length=30, num_walks=200, workers=2)
model_n2v = node2vec.fit(window=10, min_count=1)
embedding_matrix = torch.tensor([model_n2v.wv[str(i)] for i in range(adj.shape[0])]).to(args.device)

# === Step 3: Simulate temporal slices ===
T = 16
node_features_over_time = torch.stack([
    embedding_matrix + 0.01 * torch.randn_like(embedding_matrix) * t
    for t in range(T)
], dim=1)  # Shape: [N, T, F]

# === Step 4: Initialize AdiHs model ===
args.num_nodes = features.shape[0]
args.num_classes = labels.max().item() + 1
model = AdiHs(args, time_length=T).to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# === Step 5: Train AdiHs with final time step ===
for epoch in range(100):
    model.train()
    optimizer.zero_grad()

    x_final = node_features_over_time[:, -1, :]  # shape: [N, F]
    logits = model(edge_index, x=x_final, return_logits=True)
    loss = F.cross_entropy(logits[idx_train], labels[idx_train])
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(edge_index, x=x_final, return_logits=True)
        val_pred = val_logits[idx_val].max(1)[1]
        acc_val = (val_pred == labels[idx_val]).float().mean().item()
    print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val Acc: {acc_val:.4f}")

# === Step 6: Extract [N, T, F] output ===
model.eval()
temporal_outputs = []
with torch.no_grad():
    for t in range(T):
        x_t = node_features_over_time[:, t, :]
        h_t = model(edge_index=edge_index, x=x_t)
        temporal_outputs.append(h_t)
X = torch.stack(temporal_outputs, dim=1)  # [N, T, F]

# === Step 7: Apply TemporalAttentionLayer ===
att_output = model.ddy_attention_layer(X)  # [N, T, F]

# === Step 8: Flatten and Run Isolation Forest ===
X_flat = att_output.reshape(X.shape[0], -1).cpu().numpy()
clf = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
clf.fit(X_flat)
anomaly_scores = -clf.decision_function(X_flat)
anomaly_labels = clf.predict(X_flat)  # 1 = normal, -1 = anomaly

# === Step 9: Visualize or Print Anomaly Results ===
print("Top 10 Anomalous Nodes:", anomaly_scores.argsort()[-10:][::-1])

plt.figure(figsize=(8, 4))
plt.hist(anomaly_scores, bins=50)
plt.title("Anomaly Score Distribution")
plt.xlabel("Anomaly Score")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()