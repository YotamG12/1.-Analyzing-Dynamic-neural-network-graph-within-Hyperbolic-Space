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
from visualFunction import print_top_bottom_anomalous_papers, highlight_paper, plot_anomaly_score_traces, plot_temporal_anomaly_distribution, plot_temporal_sharp_changes, compute_and_plot_anomaly_scores, get_top5_anomalies_with_delta,  plot_top5_highest_delta_changes, plot_top5_trace_highest_delta_per_timestep, plot_temporal_sharp_anomaly_changes, plot_as_std_histogram,  plot_top10_std_delta_traces, plot_bottom5_lowest_delta_changes,plot_temporal_dull_anomaly_changes,plot_bottom10_std_delta_traces
import argparse

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

# === Training Loop ===
for epoch in range(args.max_epoch):
    model.train()
    optimizer.zero_grad()
    loss_total = 0
    for t in range(T):
        x_t = embedding_matrix[:, t, :].to(args.device)
        logits_t = model(edge_index, x=x_t)
        loss_t = F.cross_entropy(logits_t[idx_train], labels[idx_train])
        loss_total += loss_t
    loss_total = loss_total / T
    loss_total.backward()
    optimizer.step()

# === Step 4: Apply Attention Layer ===
model.eval()
temporal_outputs = []
with torch.no_grad():
    for t in range(T):
        x_t = embedding_matrix[:, t, :].to(args.device)
        h_t = model(edge_index, x=x_t)
        temporal_outputs.append(h_t)
X = torch.stack(temporal_outputs, dim=1)
att_output = model.ddy_attention_layer(X)  # [N, T, F]

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
    n_iters=30,
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


# === Step 11: Visualizations ===
in_degrees, out_degrees = compute_degrees(df_meta)
highlight_paper('53e9b5e0b7602d9704131ef1', df_meta, anomaly_scores, in_degrees, out_degrees, save_dir)
plot_temporal_anomaly_distribution(att_output, save_dir)
plot_temporal_sharp_changes(scores_per_time, save_dir)
plot_bottom5_lowest_delta_changes(scores_per_time, save_dir)
plot_temporal_sharp_anomaly_changes(scores_per_time, save_dir,df_meta)
plot_temporal_dull_anomaly_changes(scores_per_time, save_dir, df_meta)
compute_and_plot_anomaly_scores(att_output, df_meta, save_dir)

top5_anomalies = get_top5_anomalies_with_delta(scores_per_time, df_meta)
for anomaly in top5_anomalies:
    print(f"Top Anomaly • Node {anomaly['Node']} | Paper ID: {anomaly['Paper ID']} | Year: {anomaly['Year']} | Title: {anomaly['Title']} | Δ: {anomaly['Delta']:.4f}")

plot_as_std_histogram(scores_per_time, save_dir)
plot_top10_std_delta_traces(scores_per_time, T, save_dir, df_meta)
plot_bottom10_std_delta_traces(scores_per_time, T, save_dir, df_meta)





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

def save_table_as_png_safe(data, filename, title):
    df = pd.DataFrame(data)
    if df.empty:
        print(f"⚠️ No data to save for '{title}'. Skipping.")
        return

    try:
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
                value = str(df.iat[row, col]) if col < ncols else ""
                tbl.add_cell(row + 1, col, width, height, text=value, loc='center')

        for cell in tbl.get_celld().values():
            cell.set_fontsize(12)

        ax.add_table(tbl)
        plt.savefig(filename, bbox_inches='tight', dpi=200)
        plt.close()
        print(f"✅ Saved table: {filename}")

    except Exception as e:
        print(f"❌ Error saving table to {filename}: {e}")



# === Map time steps to real years ===
if 'year' in df_meta.columns:
    sorted_years = sorted(df_meta['year'].dropna().unique())
    if len(sorted_years) >= T:
        time_years = sorted_years[:T]
    else:
        min_year = sorted_years[0]
        time_years = [min_year + t for t in range(T)]
else:
    time_years = [f"T{t}" for t in range(T)]

# === Parameters ===
SB_min_sleep = 3       # Minimum dormant period
SB_rise_thresh = 0.8   # Threshold for awakening
FS_min_peak = 3        # Minimum active period
FS_drop_thresh = 0.2   # Threshold for falling

sleeping_beauties = []
falling_stars = []

for node in range(scores_per_time.shape[0]):
    pub_year = df_meta.loc[node, 'year']
    scores = scores_per_time[node]
    
    try:
        start_t = time_years.index(pub_year)
    except ValueError:
        continue

    # Detect Sleeping Beauty
    for t in range(start_t + SB_min_sleep, T - 1):
        dormant = scores[start_t:t]
        awakened = scores[t]
        if np.all(dormant < 0.3) and awakened > SB_rise_thresh:
            sb_years = f"{time_years[start_t]}–{time_years[t]}"
            sleeping_beauties.append({
                'Node': node,
                'Title': df_meta.loc[node, 'title'],
                'Year': pub_year,
                'Sleeping Beauty Years': sb_years
            })
            break

    # Detect Falling Star
    for t in range(start_t + FS_min_peak, T - 1):
        peak = scores[start_t:t]
        decline = scores[t]
        if np.all(peak > 0.7) and decline < FS_drop_thresh:
            fs_years = f"{time_years[start_t]}–{time_years[t]}"
            falling_stars.append({
                'Node': node,
                'Title': df_meta.loc[node, 'title'],
                'Year': pub_year,
                'Falling Star Years': fs_years
            })
            break

# === Save only top 5 entries for display ===
sleeping_beauties = sleeping_beauties[:5]
falling_stars = falling_stars[:5]

# === Save to tables ===
sb_path = save_dir / "final_sleeping_beauties_table.png"
fs_path = save_dir / "final_falling_stars_table.png"

save_table_as_png_safe(sleeping_beauties, sb_path, "Final Sleeping Beauties")
save_table_as_png_safe(falling_stars, fs_path, "Final Falling Stars")
