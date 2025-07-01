import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import networkx as nx
import sys
import os
import copy
import pickle
import ast
from torch_geometric.utils import from_networkx
from torch_geometric.utils import from_scipy_sparse_matrix
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from config import args
from model.AdiHS import AdiHs
from utilis.data_utilis import load_citation_data
from node2vec import Node2Vec
from pathlib import Path
from tabulate import tabulate
from matplotlib.table import Table
from visualFunction import print_top_bottom_anomalous_papers, highlight_paper, plot_anomaly_score_traces, plot_temporal_anomaly_distribution, plot_temporal_sharp_changes, compute_and_plot_anomaly_scores, get_top5_anomalies_with_delta,  plot_top5_highest_delta_changes, plot_top5_trace_highest_delta_per_timestep, plot_temporal_sharp_anomaly_changes, plot_as_std_histogram,  plot_top10_std_delta_traces, plot_bottom5_lowest_delta_changes,plot_temporal_dull_anomaly_changes,plot_moving_window_histograms_with_top_nodes,plot_absolute_sharp_changes
import argparse
from collections import defaultdict

save_dir = Path("./plots/anomaly_score_plots")
os.makedirs(save_dir, exist_ok=True)
sys.stdout.reconfigure(encoding='utf-8')

# === Load metadata and snapshot graphs ===
df_meta = pd.read_csv('./data/final_filtered_by_fos_and_reference.csv')
with open('./data/generate_custom_output/ind.dblpv13.snapshot_graphs', 'rb') as f:
 
    snapshot_graphs = pickle.load(f)

# === Create dynamic Node2Vec embeddings ===
T = args.Time_stamps
node2vec_embeddings = {}
for i, year in enumerate(sorted(snapshot_graphs.keys())[:T]):
    G_year = snapshot_graphs[year]
    node2vec = Node2Vec(G_year, dimensions=args.nfeat, walk_length=30, num_walks=args.num_walks, workers=args.workers)
    model_n2v = node2vec.fit(window=10, min_count=1)
    embedding = torch.zeros((len(df_meta), args.nfeat))
    for node in G_year.nodes():
        try:
            embedding[node] = torch.tensor(model_n2v.wv[str(node)])
        except KeyError:
            continue
    node2vec_embeddings[i] = embedding

embedding_matrix = torch.stack([node2vec_embeddings[t] for t in range(T)], dim=1)  # [N, T, F]

# === Step 2: Load citation data ===
adj, features, labels, idx_train, idx_val, idx_test = load_citation_data(
    dataset_str='dblpv13', use_feats=True, data_path='./data/generate_custom_output'
)

edge_index, _ = from_scipy_sparse_matrix(adj)
features = torch.FloatTensor(features.todense()).to(args.device)
labels = torch.LongTensor(labels).to(args.device)
edge_index = edge_index.to(args.device)

# === Step 3: Model ===
args.num_nodes = features.shape[0]
args.num_classes = labels.max().item() + 1
model = AdiHs(args, time_length=T).to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# === Training Loop with Overfitting Tracking ===
train_losses = []
val_accuracies = []
#epoch_anomaly_stats = []  # stores per-epoch stats for export
#node_anomaly_std_log = defaultdict(dict)  # {node_id: {epoch: std_val}}

# Fix labels shape if needed
# Make sure labels are class indices, not one-hot
# Ensure labels are a tensor
if not isinstance(labels, torch.Tensor):
    labels = torch.tensor(labels)

# Ensure labels are on the same device
labels = labels.to(args.device)

# Convert from one-hot to class indices if needed
if labels.ndim == 2 and labels.shape[1] > 1:
    labels = labels.argmax(dim=1)





for epoch in range(args.max_epoch):
    model.train()
    optimizer.zero_grad()

    # === Step 1: Forward pass through all time steps ===
    temporal_outputs = []
    for t in range(T):
        x_t = embedding_matrix[:, t, :].to(args.device)
        h_t = model(edge_index, x=x_t)
        temporal_outputs.append(h_t)

    # === Step 2: Stack and apply attention ===
    X = torch.stack(temporal_outputs, dim=1)  # [N, T, F]
    out = model.ddy_attention_layer(X)        # [N, F]
    out = out[:, -1, :]

    # === Step 3: Compute loss using attention output ===
    

    loss_total = F.cross_entropy(out[idx_train], labels[idx_train])
    loss_total.backward()
    optimizer.step()

    # === Step 4: Validation ===
    model.eval()
    with torch.no_grad():
        # Recompute X with model in eval mode
        temporal_outputs = []
        for t in range(T):
            x_t = embedding_matrix[:, t, :].to(args.device)
            h_t = model(edge_index, x=x_t)
            temporal_outputs.append(h_t)

        X_val = torch.stack(temporal_outputs, dim=1)
        att_output = model.ddy_attention_layer(X_val)
        att_output_fix = att_output[:, -1, :]
        # === Step: Record per-node average anomaly score std over time ===
    """ att_output_np = att_output.detach().cpu().numpy()  # [N, T, F]
        node_std_vals = np.std(att_output_np, axis=1)      # [N], std over time for each node

        paper_ids = df_meta['id'].values if 'id' in df_meta.columns else np.arange(len(node_std_vals))

        for node_id, paper_id, std_val in zip(range(len(node_std_vals)), paper_ids, node_std_vals):
            node_anomaly_std_log[node_id]['paper_id'] = paper_id
            node_anomaly_std_log[node_id][f'epoch {epoch} average anomaly score std'] = std_val


        records = []
        for node_id, stats in node_anomaly_std_log.items():
            row = {'node_id': node_id, 'paper_id': stats.pop('paper_id')}
            row.update(stats)
            records.append(row)

        df_anomaly_wide = pd.DataFrame(records)
        df_anomaly_wide = df_anomaly_wide.sort_values(by='node_id')
        df_anomaly_wide.to_csv("node_anomaly_std_per_epoch.csv", index=False)
        print("✅ Saved wide-format anomaly std CSV to: node_anomaly_std_per_epoch.csv")"""
        if len(idx_val) > 0 and att_output_fix.shape[0] > np.max(idx_val):
            val_pred = att_output_fix[idx_val].max(1)[1]
            acc_val = (val_pred == labels[idx_val]).float().mean().item()
        else:
            acc_val = float('nan')

    # === Step 5: Logging ===
    train_losses.append(loss_total.item())
    val_accuracies.append(acc_val)
    print(f"[Epoch {epoch}] Train Loss: {loss_total.item():.4f} | Val Accuracy: {acc_val:.4f}")
"""df_epoch_stats = pd.DataFrame(epoch_anomaly_stats)
df_epoch_stats.to_csv("epoch_node_anomaly_std.csv", index=False)
print("✅ Saved node-wise anomaly std per epoch to: epoch_node_anomaly_std.csv")"""

# === Step 4: Apply Attention Layer ===
"""model.eval()
temporal_outputs = []
with torch.no_grad():
    for t in range(T):
        x_t = embedding_matrix[:, t, :].to(args.device)
        h_t = model(edge_index, x=x_t)
        temporal_outputs.append(h_t)
X = torch.stack(temporal_outputs, dim=1)
att_output = model.ddy_attention_layer(X)  # [N, T, F]"""

# === Overfitting Diagnostic Plot ===
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss', linewidth=2)
plt.plot(val_accuracies, label='Validation Accuracy', linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Train Loss vs Validation Accuracy")
plt.legend()
plt.grid(True)
plot_path = save_dir / "loss_vs_val_acc.png"
plt.savefig(plot_path)
plt.close()
print(f"\n✅ Saved overfitting plot to: {plot_path}")

# === Step 5: MinMax Normalize and Detect Anomalies ===
X_flat = att_output.reshape(X.shape[0], -1).detach().cpu().numpy()
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_flat)
clf = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
clf.fit(X_scaled)
anomaly_scores = -clf.decision_function(X_scaled)
anomaly_labels = clf.predict(X_scaled)

# === Step 6: Optional Noise Validation ===
def validate_with_noise_injection(
    G_original, embedding_matrix, edge_index, model, T,
    idx_val, labels, n_iters=30, n_noise_nodes=10, connect_prob=0.5
):
    device = embedding_matrix.device
    N, T_actual, F = embedding_matrix.shape
    assert T == T_actual, "Mismatch in time dimension"

    # Evaluate original model on current validation set
    with torch.no_grad():
        x_val_orig = embedding_matrix[:, -1, :]
        val_logits_orig = model(edge_index, x=x_val_orig)
        if len(idx_val) > 0 and val_logits_orig.shape[0] > np.max(idx_val):
            val_pred_orig = val_logits_orig[idx_val].max(1)[1]
            acc_val_orig = (val_pred_orig == labels[idx_val]).float().mean().item()
        else:
            acc_val_orig = float('nan')
    print(f"✅ Original Model Validation Accuracy: {acc_val_orig:.4f}")

    results = []
    for it in range(n_iters):
        # Clone graph and inject synthetic noise nodes
        G_noisy = copy.deepcopy(G_original)
        original_N = G_noisy.number_of_nodes()
        new_node_ids = list(range(original_N, original_N + n_noise_nodes))

        for node in new_node_ids:
            G_noisy.add_node(node)
            for target in range(original_N):
                if np.random.rand() < connect_prob:
                    G_noisy.add_edge(node, target)
                if np.random.rand() < connect_prob:
                    G_noisy.add_edge(target, node)

        # === Add noise features ===
        noise_tensor = torch.randn((n_noise_nodes, F), device=device) * 0.1 + embedding_matrix.mean(dim=(0, 1))
        noisy_embedding_matrix = []
        for t in range(T):
            x_t_noisy = torch.cat([
                embedding_matrix[:, t, :],
                noise_tensor + 0.01 * torch.randn_like(noise_tensor) * t
            ], dim=0)
            noisy_embedding_matrix.append(x_t_noisy)
        node_features_over_time_noisy = torch.stack(noisy_embedding_matrix, dim=1)  # [N+noise, T, F]

        # Convert noisy graph to PyG format
        G_noisy_simple = nx.DiGraph()
        G_noisy_simple.add_nodes_from(G_noisy.nodes())
        G_noisy_simple.add_edges_from(G_noisy.edges())
        edge_index_noisy = from_networkx(G_noisy_simple).edge_index.to(device)

        # === Forward pass and scoring ===
        model.eval()
        with torch.no_grad():
            outputs = []
            for t in range(T):
                h_t = model(edge_index_noisy, x=node_features_over_time_noisy[:, t, :])
                outputs.append(h_t)
            X_noisy = torch.stack(outputs, dim=1)
            att_output_noisy = model.ddy_attention_layer(X_noisy)
            X_flat_noisy = att_output_noisy.reshape(X_noisy.shape[0], -1).cpu().numpy()

        # Normalize + anomaly score
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_flat_noisy)
        clf = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        clf.fit(X_scaled)
        anomaly_scores = -clf.decision_function(X_scaled)

        # Validation accuracy on noisy graph
        with torch.no_grad():
            x_val_noisy = node_features_over_time_noisy[:, -1, :]
            val_logits_noisy = model(edge_index_noisy, x=x_val_noisy)
            if len(idx_val) > 0 and val_logits_noisy.shape[0] > np.max(idx_val):
                val_pred_noisy = val_logits_noisy[idx_val].max(1)[1]
                acc_val_noisy = (val_pred_noisy == labels[idx_val]).float().mean().item()
            else:
                acc_val_noisy = float('nan')

        # Separate scores
        original_scores = anomaly_scores[:original_N]
        noise_scores = anomaly_scores[original_N:]
        frac_above_95 = (noise_scores > np.percentile(original_scores, 95)).mean()

        print(f"[Validation Iter {it}] Mean Anomaly: Original={original_scores.mean():.4f} | Noise={noise_scores.mean():.4f} | Frac>95%={frac_above_95:.2f} | Acc Noisy={acc_val_noisy:.4f}")
        results.append({
            'iteration': it,
            'mean_anomaly_original': float(np.mean(original_scores)),
            'mean_anomaly_noise': float(np.mean(noise_scores)),
            'std_anomaly_noise': float(np.std(noise_scores)),
            'frac_above_95': float(frac_above_95),
            'val_acc_noisy': float(acc_val_noisy),
        })

    return results


validation_results = validate_with_noise_injection(
    G_original=snapshot_graphs[sorted(snapshot_graphs.keys())[0]],
    embedding_matrix=embedding_matrix,
    edge_index=edge_index,
    model=model,
    T=T,
    idx_val=idx_val,
    labels=labels,
    n_iters=args.validation_iteration,
    n_noise_nodes=10,
    connect_prob=0.5
)


# === Continue with visualizations ===

# === Step 10: Temporal Scoring Per Time Step ===
scores_per_time = []
for t in range(T):
    vecs_t = att_output[:, t, :].detach().cpu().numpy()
    clf_t = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    clf_t.fit(vecs_t)
    scores_t = -clf_t.decision_function(vecs_t)
    scores_per_time.append(scores_t)
scores_per_time = np.stack(scores_per_time, axis=1)



def compute_degrees(df_meta):
    """Compute in-degree and out-degree lists for all nodes (to be used in scatter plots)."""
    in_degrees = []
    out_degrees = []

    # Map paper id to index
    id_to_index = {pid: idx for idx, pid in enumerate(df_meta['id'])}

    for i, row in df_meta.iterrows():
        refs = row.get('references', '[]')
        try:
            refs = ast.literal_eval(refs) if isinstance(refs, str) else refs
        except Exception:
            refs = []

        out_degrees.append(len(refs))  # cites count

        # count how many cite this paper (in-degree)
        in_deg = 0
        for other_refs in df_meta['references']:
            try:
                parsed = ast.literal_eval(other_refs) if isinstance(other_refs, str) else other_refs
                if row['id'] in parsed:
                    in_deg += 1
            except:
                continue
        in_degrees.append(in_deg)

    return in_degrees, out_degrees

N, T, F = att_output.shape
scores_per_time = []

for t in range(T):
        # extract features at time t
        vecs_t = att_output[:, t, :].detach().cpu().numpy()
        # fit a fresh IsolationForest
        clf_t = IsolationForest(n_estimators=100,
                                contamination=0.05,
                                random_state=42)
        clf_t.fit(vecs_t)
        # decision_function: negative means more anomalous → take minus
        scores_t = -clf_t.decision_function(vecs_t)
        scores_per_time.append(scores_t)

scores_per_time = np.stack(scores_per_time, axis=1)  # shape [N, T]

    # Identify Top-5 Papers by Avg Anomaly
avg_scores = scores_per_time.mean(axis=1)          # [N]
top5_idx = np.argsort(avg_scores)[-5:]             # indices of top 5 (most anomalous)
bottom5_idx = np.argsort(avg_scores)[:5]           # indices of bottom 5 (least anomalous)

# === Step 11: Visualizations ===
"""in_degrees, out_degrees = compute_degrees(df_meta)
highlight_paper('53e9b5e0b7602d9704131ef1', df_meta, anomaly_scores, in_degrees, out_degrees, save_dir)"""
plot_temporal_anomaly_distribution(att_output, save_dir)
#plot_temporal_sharp_changes(scores_per_time, save_dir)
#plot_bottom5_lowest_delta_changes(scores_per_time, save_dir)
plot_temporal_sharp_anomaly_changes(scores_per_time, save_dir,df_meta)
#plot_temporal_dull_anomaly_changes(scores_per_time, save_dir, df_meta)
#compute_and_plot_anomaly_scores(att_output, df_meta, save_dir)

top5_anomalies = get_top5_anomalies_with_delta(scores_per_time, df_meta)
for anomaly in top5_anomalies:
    print(f"Top Anomaly • Node {anomaly['Node']} | Paper ID: {anomaly['Paper ID']} | Year: {anomaly['Year']} | Title: {anomaly['Title']} | Δ: {anomaly['Delta']:.4f}")

plot_as_std_histogram(scores_per_time, save_dir)
#plot_top10_std_delta_traces(scores_per_time, T, save_dir, df_meta)
#plot_bottom10_std_delta_traces(scores_per_time, T, save_dir, df_meta)
#detect_and_plot_sleeping_beauties_by_delta(scores_per_time, df_meta, save_dir)
#detect_and_plot_falling_stars_by_delta(scores_per_time, df_meta, save_dir)
plot_moving_window_histograms_with_top_nodes(
    att_output,  # shape [N, T]
    window_size=10,
    step_size=5,
    save_dir=Path("plots/moving_histograms")
)
plot_absolute_sharp_changes(scores_per_time, save_dir=Path("plots"), df_meta=df_meta)














