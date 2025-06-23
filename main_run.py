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
from torch_geometric.utils import from_networkx
from torch_geometric.utils import from_scipy_sparse_matrix
from sklearn.ensemble import IsolationForest
from config import args
from model.AdiHS import AdiHs
from utilis.data_utilis import load_citation_data
from node2vec import Node2Vec
from pathlib import Path
from tabulate import tabulate
from matplotlib.table import Table
from visualFunction import print_top_bottom_anomalous_papers, highlight_paper, plot_anomaly_score_traces, plot_temporal_anomaly_distribution, plot_temporal_sharp_changes, compute_and_plot_anomaly_scores, get_top5_anomalies_with_delta,  plot_top5_highest_delta_changes, plot_top5_trace_highest_delta_per_timestep, plot_temporal_sharp_anomaly_changes, plot_as_std_histogram,  plot_top10_std_delta_traces
import argparse

save_dir = Path("./plots/anomaly_score_plots")  # Shared directory for all plots
os.makedirs(save_dir, exist_ok=True)

sys.stdout.reconfigure(encoding='utf-8')

# === Step 1: Load citation data ===
adj, features, labels, idx_train, idx_val, idx_test = load_citation_data(
    dataset_str='dblpv13', use_feats=True, data_path='./data/generate_custom_output'
)


edge_index, _ = from_scipy_sparse_matrix(adj)
features = torch.FloatTensor(features.todense()).to(args.device)
labels = torch.LongTensor(labels).to(args.device)
edge_index = edge_index.to(args.device)

# === Step 2: Node2Vec Embedding ===
G = nx.from_scipy_sparse_array(adj)
node2vec = Node2Vec(G, dimensions=args.nfeat, walk_length=30, num_walks=args.num_walks, workers=args.workers)
model_n2v = node2vec.fit(window=10, min_count=1)
embedding_matrix = torch.tensor(
    np.array([model_n2v.wv[str(i)] for i in range(adj.shape[0])])
).to(args.device)

# === Step 3: Year-based Temporal Grouping (UI-driven bins) ===
df_meta = pd.read_csv('./data/final_filtered_by_fos_and_reference.csv')
df_years = df_meta['year'].values
min_year, max_year = df_years.min(), df_years.max()
T = args.Time_stamps  # Use Time_stamps value from config
num_epochs = args.max_epoch  # Use max_epoch for number of epochs

bins = list(range(min_year, max_year + 2))
node_time_step = np.digitize(df_years, bins) - 1



# === Step 4: Simulate temporal slices ===
node_features_over_time = torch.stack([
    embedding_matrix + 0.01 * torch.randn_like(embedding_matrix) * t
    for t in range(T)
], dim=1)  # [N, T, F]

# === Step 5: Initialize Model ===
args.num_nodes = features.shape[0]
args.num_classes = labels.max().item() + 1
model = AdiHs(args, time_length=T).to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)



# === Training Loop ===
for epoch in range(num_epochs):  # Use epochs value from UI
    model.train()
    optimizer.zero_grad()
    loss_total = 0
    for t in range(T):
        x_t = node_features_over_time[:, t, :]
        logits_t = model(edge_index, x=x_t)
        loss_t = F.cross_entropy(logits_t[idx_train], labels[idx_train])
        loss_total += loss_t
    loss_total = loss_total / T  # Average loss across time steps
    loss_total.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        x_val = node_features_over_time[:, -1, :]
        val_logits = model(edge_index, x=x_val)
        val_pred = val_logits[idx_val].max(1)[1]
        acc_val = (val_pred == labels[idx_val]).float().mean().item()
    print(f"Epoch {epoch:03d} | Loss: {loss_total.item():.4f} | Val Acc: {acc_val:.4f}")

# === Step 7: Extract [N, T, F] embeddings ===
model.eval()
temporal_outputs = []
with torch.no_grad():
    for t in range(T):
        x_t = node_features_over_time[:, t, :]
        h_t = model(edge_index=edge_index, x=x_t)
        temporal_outputs.append(h_t)
X = torch.stack(temporal_outputs, dim=1)  # [N, T]

# === Step 8: Apply Temporal Attention Layer ===
att_output = model.ddy_attention_layer(X)  # [N, T, F]
X_flat = att_output.reshape(X.shape[0], -1).detach().cpu().numpy()

# === Step 9: Anomaly Detection ===
clf = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
clf.fit(X_flat)
anomaly_scores = -clf.decision_function(X_flat)
anomaly_labels = clf.predict(X_flat)



# === Graphical Analysis ===
G = nx.from_scipy_sparse_array(adj)
DG = G.to_directed()
degrees = np.array([G.degree(i) for i in range(G.number_of_nodes())])
in_degrees = np.array([DG.in_degree(i) for i in range(G.number_of_nodes())])
out_degrees = np.array([DG.out_degree(i) for i in range(G.number_of_nodes())])




# === Step 12: Iterative Validation with Artificial Noise Injection ===
def validate_with_noise_injection(
    G_original, embedding_matrix, node_features_over_time, edge_index, model, T, n_iters=30, n_noise_nodes=10, connect_prob=0.5
):
    
    """Iteratively injects artificial noise (random nodes/edges) into the network and compares anomaly scores before and after.
    Args:
        G_original: Original networkx graph
        embedding_matrix: Original node embeddings (torch.Tensor)
        node_features_over_time: [N, T, F] tensor
        edge_index: PyG edge_index
        model: Trained model
        T: Number of time steps
        n_iters: Number of validation iterations
        n_noise_nodes: Number of random nodes to add per iteration
        connect_prob: Probability to connect a noise node to an existing node"""
    
    # Compute original model validation accuracy (before noise) ONCE before the loop
    with torch.no_grad():
        x_val_orig = node_features_over_time[:, -1, :]
        val_logits_orig = model(edge_index, x=x_val_orig)
        if len(idx_val) > 0 and val_logits_orig.shape[0] > np.max(idx_val):
            val_pred_orig = val_logits_orig[idx_val].max(1)[1]
            acc_val_orig = (val_pred_orig == labels[idx_val]).float().mean().item()
        else:
            acc_val_orig = float('nan')
    print(f"Original Model Validation Accuracy: {acc_val_orig:.4f}")

    results = []
    for it in range(n_iters):
        # 1. Copy the original graph
        G_noisy = copy.deepcopy(G_original)
        N = G_noisy.number_of_nodes()
        new_node_ids = list(range(N, N + n_noise_nodes))
        # 2. Add random nodes (isolated or with random edges)
        for node in new_node_ids:
            G_noisy.add_node(node)
            for target in range(N):
                if np.random.rand() < connect_prob:
                    G_noisy.add_edge(node, target)
                if np.random.rand() < connect_prob:
                    G_noisy.add_edge(target, node)
        # 3. Generate new embeddings for new nodes (random or mean)
        F = embedding_matrix.shape[1]
        new_embeds = torch.randn(n_noise_nodes, F).to(embedding_matrix.device) * 0.1 + embedding_matrix.mean(0)
        embedding_matrix_noisy = torch.cat([embedding_matrix, new_embeds], dim=0)
        # 4. Simulate temporal slices for noisy graph
        node_features_over_time_noisy = torch.stack([
            embedding_matrix_noisy + 0.01 * torch.randn_like(embedding_matrix_noisy) * t
            for t in range(T)
        ], dim=1)
        # 5. Update edge_index
        # Remove all edge attributes to avoid ValueError in from_networkx
        G_noisy_simple = nx.DiGraph()
        G_noisy_simple.add_nodes_from(G_noisy.nodes())
        G_noisy_simple.add_edges_from(G_noisy.edges())
        data_noisy = from_networkx(G_noisy_simple)
        edge_index_noisy = data_noisy.edge_index.to(embedding_matrix.device)
        # 6. Get anomaly scores for noisy graph
        model.eval()
        temporal_outputs = []
        with torch.no_grad():
            for t in range(T):
                x_t = node_features_over_time_noisy[:, t, :]
                h_t = model(edge_index_noisy, x=x_t)
                temporal_outputs.append(h_t)
        X_noisy = torch.stack(temporal_outputs, dim=1)
        att_output_noisy = model.ddy_attention_layer(X_noisy)
        X_flat_noisy = att_output_noisy.reshape(X_noisy.shape[0], -1).detach().cpu().numpy()
        clf_noisy = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        clf_noisy.fit(X_flat_noisy)
        anomaly_scores_noisy = -clf_noisy.decision_function(X_flat_noisy)
        # 6a. Validation accuracy for noisy graph (on original validation nodes)
        with torch.no_grad():
            x_val_noisy = node_features_over_time_noisy[:, -1, :]
            val_logits_noisy = model(edge_index_noisy, x=x_val_noisy)
            if len(idx_val) > 0 and val_logits_noisy.shape[0] > np.max(idx_val):
                val_pred_noisy = val_logits_noisy[idx_val].max(1)[1]
                acc_val_noisy = (val_pred_noisy == labels[idx_val]).float().mean().item()
            else:
                acc_val_noisy = float('nan')
        # 7. Compare anomaly scores (focus on original nodes)
        anomaly_scores_original = anomaly_scores_noisy[:N]
        noise_scores = anomaly_scores_noisy[N:]
        # 8. Quantitative analysis: mean/std/max of anomaly scores before/after
        result = {
            'iteration': it,
            'mean_anomaly_original': float(np.mean(anomaly_scores_original)),
            'std_anomaly_original': float(np.std(anomaly_scores_original)),
            'max_anomaly_original': float(np.max(anomaly_scores_original)),
            'mean_anomaly_noise': float(np.mean(noise_scores)),
            'std_anomaly_noise': float(np.std(noise_scores)),
            'max_anomaly_noise': float(np.max(noise_scores)),
            'val_acc_noisy': acc_val_noisy
        }
        # Print anomaly score comparison for noise nodes
        frac_above_95 = (noise_scores > np.percentile(anomaly_scores_original, 95)).mean()
        print(f"[Validation Iter {it}] Original mean: {result['mean_anomaly_original']:.4f}, Noise mean: {result['mean_anomaly_noise']:.4f}, Model Accuracy: {acc_val_orig:.4f}, Noisy Validation Accuracy: {acc_val_noisy:.4f}")
        print(f"Noise nodes mean anomaly score: {result['mean_anomaly_noise']:.4f}")
        print(f"Original nodes mean anomaly score: {result['mean_anomaly_original']:.4f}")
        print(f"Fraction of noise nodes above 95th percentile of original: {frac_above_95:.2f}")
        results.append(result)
    return results

# Run validation with noise injection
validation_results = validate_with_noise_injection(
    G, embedding_matrix, node_features_over_time, edge_index, model, T,
    n_iters=30, n_noise_nodes=10, connect_prob=0.5
)


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



# === Map time steps to real years ===
if 'year' in df_meta.columns:
    sorted_years = sorted(df_meta['year'].dropna().unique())
    if len(sorted_years) >= T:
        time_years = sorted_years[:T]
    else:
        time_years = [min_year + t for t in range(T)]
else:
    time_years = [f"T{t}" for t in range(T)]

# === Step: Detect Sleeping Beauty and Falling Star based on anomaly score trends ===
SB_min_sleep = 3
SB_rise_std = 2
FS_min_fall = 3
FS_drop_std = 2

sleeping_beauties = []
falling_stars = []

for node in range(scores_per_time.shape[0]):
    pub_year = df_meta.loc[node, 'year']
    scores = scores_per_time[node]
    norm_scores = (scores - scores.mean()) / (scores.std() + 1e-8)

    
    try:
        start_t = time_years.index(pub_year)
    except ValueError:
        continue  # year not in time_years range

    for t in range(start_t + SB_min_sleep, T - 1):
        if np.all(norm_scores[start_t:t] < 0.5) and norm_scores[t] > SB_rise_std:
            sb_years = f"{time_years[start_t]}–{time_years[t]}"
            sleeping_beauties.append({
                'Node': node,
                'Title': df_meta.loc[node, 'title'],
                'Year': pub_year,
                'Sleeping Beauty Years': sb_years
            })
            break

    for t in range(start_t + FS_min_fall, T - 1):
        if np.all(norm_scores[start_t:t] > 0.5) and norm_scores[t] < -FS_drop_std:
            fs_years = f"{time_years[start_t]}–{time_years[t]}"
            falling_stars.append({
                'Node': node,
                'Title': df_meta.loc[node, 'title'],
                'Year': pub_year,
                'Falling Star Years': fs_years
            })
            break

# === Save only top 5 entries for readability ===
sleeping_beauties = sleeping_beauties[:5]
falling_stars = falling_stars[:5]

# === Function: Save a table as PNG safely ===
def save_table_as_png_safe(data, filename, title):
    df = pd.DataFrame(data)
    if df.empty:
        print(f"⚠️ No data to save for '{title}'. Skipping.")
        return

    fig, ax = plt.subplots(figsize=(16, 3 + len(df) * 0.7))
    ax.set_axis_off()
    tbl = Table(ax, bbox=[0, 0, 1, 1])
    nrows, ncols = df.shape
    width, height = 1.0 / ncols, 1.0 / (nrows + 1)

    ax.set_title(title, fontsize=20, fontweight="bold", pad=30)

    for i, column in enumerate(df.columns):
        tbl.add_cell(0, i, width, height, text=column, loc='center', facecolor='lightgrey')
    for row in range(nrows):
        for col in range(ncols):
            tbl.add_cell(row+1, col, width, height, text=str(df.iat[row, col]), loc='center')

    for cell in tbl.get_celld().values():
        cell.set_fontsize(12)

    ax.add_table(tbl)
    plt.savefig(filename, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"✅ Saved table: {filename}")

# === Save PNG tables for real detected anomaly patterns ===
sb_path = save_dir / "final_sleeping_beauties_table.png"
fs_path = save_dir / "final_falling_stars_table.png"

save_table_as_png_safe(sleeping_beauties, sb_path, "Final Sleeping Beauties")
save_table_as_png_safe(falling_stars, fs_path, "Final Falling Stars")

# Call functions for printing and plotting
highlight_paper('53e9b5e0b7602d9704131ef1', df_meta, anomaly_scores, in_degrees, out_degrees, save_dir)
plot_temporal_anomaly_distribution(att_output, save_dir)
plot_temporal_sharp_changes(scores_per_time, save_dir)
plot_temporal_sharp_anomaly_changes(scores_per_time, save_dir)

# Removed computation and plotting of anomaly scores and replaced with call to visualFunction
compute_and_plot_anomaly_scores(att_output, df_meta, save_dir)

# Compute top 5 anomalies with sharp delta anomaly scores
scores_per_time = np.stack(scores_per_time, axis=1)  # Ensure scores_per_time is stacked

top5_anomalies = get_top5_anomalies_with_delta(scores_per_time, df_meta)
print("\nTop 5 Anomalies with Sharp Delta Anomaly Scores:")
for anomaly in top5_anomalies:
    print(f"  • Node {anomaly['Node']} | Paper ID: {anomaly['Paper ID']} | Year: {anomaly['Year']} | Title: {anomaly['Title']} | Delta: {anomaly['Delta']:.4f}")


"""plot_top5_highest_delta_changes(scores_per_time, save_dir)
plot_top5_trace_highest_delta_per_timestep(scores_per_time, save_dir)"""

"""# Load metadata
if os.path.exists('./data/final_filtered_by_fos_and_reference.csv'):
    df_meta = pd.read_csv('./data/final_filtered_by_fos_and_reference.csv')
    if 'year' in df_meta.columns:
        df_years = df_meta['year'].values
        min_year, max_year = df_years.min(), df_years.max()
    else:
        raise ValueError("Year column not found in metadata.")
else:
    raise FileNotFoundError("Metadata file not found.")"""

# Call the new plot_as_std_histogram function to generate the histogram of AS standard deviations for all nodes.
plot_as_std_histogram(scores_per_time, save_dir)
print("📐 scores_per_time shape:", scores_per_time.shape)

plot_top10_std_delta_traces(scores_per_time, T, save_dir)