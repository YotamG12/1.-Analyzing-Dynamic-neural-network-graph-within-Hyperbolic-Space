import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np
from collections import Counter

def compute_and_plot_anomaly_scores(att_output, df_meta, save_dir):
    """Computes per-time-step anomaly scores and plots top/bottom anomalous papers."""
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

    # Print metadata
    print_top_bottom_anomalous_papers(top5_idx, bottom5_idx, df_meta)

    # Plot Anomaly Score Traces for Top-5
    plot_anomaly_score_traces(top5_idx, bottom5_idx, scores_per_time, T, save_dir)

def print_top_bottom_anomalous_papers(top5_idx, bottom5_idx, df_meta):
    """Prints metadata for top-5 and bottom-5 anomalous papers."""
    print("\nTop 5 Anomalous Papers (by average score):")
    for idx in top5_idx:
        pid   = df_meta.loc[idx, 'id'] if 'id' in df_meta.columns else idx
        title = df_meta.loc[idx, 'title'] if 'title' in df_meta.columns else 'Unknown'
        year  = df_meta.loc[idx, 'year'] if 'year' in df_meta.columns else 'Unknown'
        print(f"  • Node {idx} | Paper ID: {pid} | Year: {year} | Title: {title}")

    print("\nTop 5 Least Anomalous Papers (by average score):")
    for idx in bottom5_idx:
        pid   = df_meta.loc[idx, 'id'] if 'id' in df_meta.columns else idx
        title = df_meta.loc[idx, 'title'] if 'title' in df_meta.columns else 'Unknown'
        year  = df_meta.loc[idx, 'year'] if 'year' in df_meta.columns else 'Unknown'
        print(f"  • Node {idx} | Paper ID: {pid} | Year: {year} | Title: {title}")

def plot_anomaly_score_traces(top5_idx, bottom5_idx, scores_per_time, T, save_dir):
    """Plots anomaly score traces for top-5 and bottom-5 anomalous papers."""
    # Top-5
    plt.figure(figsize=(10, 6))
    for idx in top5_idx:
        plt.plot(range(T),
                 scores_per_time[idx],
                 marker='o',
                 label=f'Node {idx}')
    plt.xlabel("Time Step")
    plt.ylabel("Anomaly Score")
    plt.title("Anomaly Score Over Time for Top 5 Anomalous Papers")
    plt.legend()
    plt.grid(True)
    trace_path_top = save_dir / "top5_anomaly_traces.png"
    plt.savefig(trace_path_top)
    plt.close()
    print(f"\nSaved top-5 anomaly trace plot to: {trace_path_top}")

    # Bottom-5
    plt.figure(figsize=(10, 6))
    for idx in bottom5_idx:
        plt.plot(range(T),
                 scores_per_time[idx],
                 marker='x',
                 linestyle='--',
                 label=f'Node {idx}')
    plt.xlabel("Time Step")
    plt.ylabel("Anomaly Score")
    plt.title("Anomaly Score Over Time for Top 5 Least Anomalous Papers")
    plt.legend()
    plt.grid(True)
    trace_path_bottom = save_dir / "top5_anomaly_traces.png"
    plt.savefig(trace_path_bottom)
    plt.close()
    print(f"Saved bottom-5 anomaly trace plot to: {trace_path_bottom}")

def plot_temporal_sharp_changes(scores_per_time, save_dir):
    """Plots temporal sharp changes in anomaly scores for top nodes."""
    as_diff = np.diff(scores_per_time, axis=1)  # shape [N, T-1]
    mean_diff = as_diff.mean()
    std_diff = as_diff.std()

    # Identify nodes with sharp changes (any time step with |diff| > mean+2*std)
    sharp_change_nodes = []
    sharp_change_times = []
    for node in range(as_diff.shape[0]):
        for t in range(as_diff.shape[1]):
            if abs(as_diff[node, t]) > mean_diff + 2 * std_diff:
                sharp_change_nodes.append(node)
                sharp_change_times.append(t)

    # Plot: For a few nodes with the sharpest changes, show their anomaly score trace and mark the sharp change
    if sharp_change_nodes:
        # Count sharp changes per node and pick top 5 nodes with most sharp changes
        node_counts = Counter(sharp_change_nodes)
        top_nodes = [n for n, _ in node_counts.most_common(5)]
        plt.figure(figsize=(12, 7))
        for node in top_nodes:
            plt.plot(range(scores_per_time.shape[1]), scores_per_time[node], label=f'Node {node}')
            # Mark sharp change points
            for t in np.where(abs(as_diff[node]) > mean_diff + 2 * std_diff)[0]:
                plt.scatter(t+1, scores_per_time[node, t+1], color='red', s=60, zorder=5)
        plt.xlabel("Time Step")
        plt.ylabel("Anomaly Score")
        plt.title("Temporal Sharp Changes in Anomaly Score (Top Nodes)")
        plt.legend()
        plt.grid(True)
        trace_path_sharp = save_dir / "temporal_sharp_changes.png"
        plt.savefig(trace_path_sharp)
        plt.close()
        print(f"Saved temporal sharp change plot to: {trace_path_sharp}")
    else:
        print("No sharp temporal changes detected in anomaly scores.")

def plot_temporal_sharp_anomaly_changes(scores_per_time, save_dir):
    """
    Plots the temporal sharp changes in anomaly scores for the 5 nodes with the largest sharp changes (highest max delta).
    Each line represents a node, and sharp change points are marked.
    """
    as_diff = np.diff(scores_per_time, axis=1)  # shape [N, T-1]
    mean_diff = as_diff.mean()
    std_diff = as_diff.std()

    # Compute the maximum sharp change for each node
    sharpness = np.max(np.abs(as_diff), axis=1)
    # Get the indices of the top 5 nodes with the largest sharp change
    top5_nodes = np.argsort(sharpness)[-5:]

    plt.figure(figsize=(12, 7))
    for node in top5_nodes:
        plt.plot(range(scores_per_time.shape[1]), scores_per_time[node], label=f'Node {node}')
        # Mark sharp change points
        for t in np.where(np.abs(as_diff[node]) > mean_diff + 2 * std_diff)[0]:
            plt.scatter(t+1, scores_per_time[node, t+1], color='red', s=60, zorder=5)
    plt.xlabel("Time Step")
    plt.ylabel("Anomaly Score")
    plt.title("Temporal Sharp Changes in Anomaly Score (Top 5 Most Extreme Nodes)")
    plt.legend()
    plt.grid(True)
    plot_path = save_dir / "temporal_sharp_anomaly_changes.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved temporal sharp anomaly change plot to: {plot_path}")

def plot_temporal_anomaly_distribution(att_output, save_dir):
    """Plots histogram of anomaly score distribution across time steps."""
    N, T = att_output.shape[:2]
    for t in range(T):
        time_step_vectors = att_output[:, t, :].detach().cpu().numpy()
        clf_t = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        clf_t.fit(time_step_vectors)
        scores = -clf_t.decision_function(time_step_vectors)

        plt.figure()
        plt.hist(scores, bins=50, alpha=0.7, edgecolor='black')
        plt.title(f"Anomaly Score Distribution at Time Step {t}")
        plt.xlabel("Anomaly Score")
        plt.ylabel("Number of Nodes")
        plt.grid(True)
        plot_path = save_dir / f"anomaly_hist_t{t}.png"
        plt.savefig(plot_path)
        plt.close()

def save_table_as_png_safe(data, filename, title):
    """Saves tabular data as a PNG image."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=data, colLabels=['Node', 'Title', 'Year', 'Years'], loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.title(title)
    plt.savefig(filename)
    plt.close()

def highlight_paper(paper_id, df_meta, anomaly_scores, in_degrees, out_degrees, save_dir):
    """Highlights a specific paper with detailed metadata."""
    if paper_id not in df_meta['id'].values:
        print(f"Paper ID '{paper_id}' not found.")
        return

    node_id = df_meta.index[df_meta['id'] == paper_id].item()
    score = anomaly_scores[node_id]
    title = df_meta.loc[node_id, 'title'] if 'title' in df_meta.columns else 'Unknown Title'
    year = df_meta.loc[node_id, 'year'] if 'year' in df_meta.columns else 'Unknown Year'

    # --- Fix reference parsing for in-degree and out-degree ---
    # Parse references column if it's a string representation of a list
    def parse_refs(refs):
        if isinstance(refs, list):
            return refs
        if isinstance(refs, str):
            try:
                import ast
                parsed = ast.literal_eval(refs)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                return []
        return []

    # Out-degree (cites): how many papers this paper cites
    paper_references = parse_refs(df_meta.loc[node_id, 'references']) if 'references' in df_meta.columns else []
    cites_count = len(paper_references)

    # In-degree (cited by): how many papers cite this paper
    cited_by_count = 0
    if 'references' in df_meta.columns:
        for refs in df_meta['references']:
            ref_list = parse_refs(refs)
            if paper_id in ref_list:
                cited_by_count += 1

    print(f"\n📄 Tracking Paper ID: {paper_id}")
    print(f"Node ID: {node_id}")
    print(f"Title: {title}")
    print(f"Year: {year}")
    print(f"Anomaly Score: {score:.4f}")
    print(f"In-Degree (Cited by): {cited_by_count}")
    print(f"Out-Degree (Cites): {cites_count}")

    # In-Degree Plot
    plt.figure()
    plt.scatter(in_degrees, anomaly_scores, alpha=0.3)
    plt.scatter(cited_by_count, score, color='red')
    plt.title(f"{title} ({year})\nAnomaly vs. In-Degree\nCited by {cited_by_count} papers")
    plt.xlabel("In-Degree")
    plt.ylabel("Anomaly Score")
    plt.grid(True)
    plot_path_in = save_dir / f"{paper_id}_in_degree.png"
    plt.savefig(plot_path_in)
    plt.close()

    # Out-Degree Plot
    plt.figure()
    plt.scatter(out_degrees, anomaly_scores, alpha=0.3, color='orange')
    plt.scatter(cites_count, score, color='blue')
    plt.title(f"{title} ({year})\nAnomaly vs. Out-Degree\nCites {cites_count} papers")
    plt.xlabel("Out-Degree")
    plt.ylabel("Anomaly Score")
    plt.grid(True)
    plot_path_out = save_dir / f"{paper_id}_out_degree.png"
    plt.savefig(plot_path_out)
    plt.close()

def get_top5_anomalies_with_delta(scores_per_time, df_meta):
    """Returns top 5 anomalies with the sharpest delta in anomaly scores and their deltas."""
    deltas = np.abs(np.diff(scores_per_time, axis=1))  # Compute deltas across time steps
    max_deltas = deltas.max(axis=1)  # Maximum delta for each node
    top5_idx = np.argsort(max_deltas)[-5:]  # Indices of top 5 nodes with sharpest deltas

    top5_anomalies = []
    for idx in top5_idx:
        pid = df_meta.loc[idx, 'id'] if 'id' in df_meta.columns else idx
        title = df_meta.loc[idx, 'title'] if 'title' in df_meta.columns else 'Unknown'
        year = df_meta.loc[idx, 'year'] if 'year' in df_meta.columns else 'Unknown'
        delta = max_deltas[idx]
        top5_anomalies.append({
            'Node': idx,
            'Paper ID': pid,
            'Year': year,
            'Title': title,
            'Delta': delta
        })

    return top5_anomalies

def plot_top5_highest_delta_changes(scores_per_time, save_dir):
    """Plots a line graph for the top 5 nodes with the highest change in delta anomaly scores."""
    deltas = np.abs(np.diff(scores_per_time, axis=1))  # Compute deltas across time steps
    max_deltas = deltas.max(axis=1)  # Maximum delta for each node
    top5_idx = np.argsort(max_deltas)[-5:]  # Indices of top 5 nodes with highest deltas

    plt.figure(figsize=(10, 6))
    for idx in top5_idx:
        plt.plot(range(1, deltas.shape[1] + 1), deltas[idx], label=f"Node {idx}")

    plt.xlabel("Time Steps")
    plt.ylabel("Delta Change")
    plt.title("Top 5 Nodes with Highest Change in Delta Anomaly Scores")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_path = save_dir / "top5_highest_delta_changes.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"✅ Line graph saved at: {plot_path}")

def plot_top5_trace_highest_delta_per_timestep(scores_per_time, save_dir):
    """
    For each time step, find the node with the highest delta anomaly score (change from previous time step),
    then plot the trace (delta over time) for the top 5 unique nodes that appear most frequently as the highest delta node.
    X-axis: time steps (range T), Y-axis: delta change in anomaly score.
    """
    deltas = np.abs(np.diff(scores_per_time, axis=1))  # shape [N, T-1]
    T = scores_per_time.shape[1]
    # For each time step, find the node with the highest delta
    top_nodes_per_t = np.argmax(deltas, axis=0)  # shape [T-1]
   
    node_counts = Counter(top_nodes_per_t)
    top5_nodes = [node for node, _ in node_counts.most_common(5)]

    plt.figure(figsize=(10, 6))
    for idx in top5_nodes:
        # Pad the delta array to align with T time steps (first delta is 0)
        delta_trace = np.insert(deltas[idx], 0, 0)
        plt.plot(range(T), delta_trace, label=f"Node {idx}")

    plt.xlabel("Time Steps")
    plt.ylabel("Delta Change in Anomaly Score")
    plt.title("Trace of Top 5 Nodes with Highest Delta Anomaly Score per Time Step")
    plt.legend()
    plt.grid(True)
    plot_path = save_dir / "top5_trace_highest_delta_per_timestep.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"✅ Trace plot saved at: {plot_path}")

def plot_as_std_histogram(scores_per_time, save_dir):
    """
    Plots a histogram of the standard deviations of anomaly score (AS) vectors over time for all nodes.
    Each node's std is calculated across time steps.
    """
    stds = np.std(scores_per_time, axis=1)  # shape [N]
    plt.figure(figsize=(10, 6))
    plt.hist(stds, bins=40, color='skyblue', edgecolor='black', alpha=0.8)
    plt.xlabel("Standard Deviation of Anomaly Score (AS) Over Time")
    plt.ylabel("Number of Nodes")
    plt.title("Histogram of AS Standard Deviations Across Time (All Nodes)")
    plt.grid(True)
    plot_path = save_dir / "as_std_histogram.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved AS std histogram to: {plot_path}")

def plot_top10_std_delta_traces(scores_per_time, T, save_dir):
    """
    Plots delta anomaly score traces over time for the top 10 nodes with the highest standard deviation.
    X-axis shows time slices [0, T-1].
    """
    scores_per_time = scores_per_time.T  # ensure shape is [N, T]

    stds = np.std(scores_per_time, axis=1)
    n_nodes = scores_per_time.shape[0]
    n_top = min(10, n_nodes)  # top 10, or less if fewer nodes
    top_indices = np.argsort(stds)[-n_top:]
    
    deltas = np.diff(scores_per_time, axis=1)  # shape: [N, T-1]

    plt.figure(figsize=(14, 8))
    for idx in top_indices:
        delta_trace = deltas[idx]
        if delta_trace.shape[0] != T - 1:
            print(f"❌ Mismatch in delta shape for node {idx}: {delta_trace.shape}")
            continue
        delta_trace = np.insert(delta_trace, 0, 0)  # pad to length T
        plt.plot(np.arange(T), delta_trace, label=f'Node {idx}', alpha=0.7)

    plt.xlabel("Time Slice")
    plt.ylabel("Delta Anomaly Score (AS)")
    plt.title("Delta AS Traces for Top 10 Nodes with Highest AS Std")
    plt.xlim(0, T - 1)
    plt.grid(True)
    plt.legend()
    plot_path = save_dir / f"top10_std_delta_traces.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"✅ Saved delta AS traces for top 10 nodes to: {plot_path}")

