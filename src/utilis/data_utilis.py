import os
import pickle as pkl
import sys
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

def load_citation_data(dataset_str, use_feats, data_path, split_seed=None):
    """
    Load citation network data and features for a given dataset.

    Args:
        dataset_str (str): Dataset name (e.g., 'dblpv13').
        use_feats (bool): Whether to use node features or identity matrix.
        data_path (str): Path to data directory containing citation files.
        split_seed (int, optional): Random seed for splitting (unused).

    Returns:
        tuple: (adj, features, labels, idx_train, idx_val, idx_test)
            adj (scipy.sparse matrix): Adjacency matrix of the citation graph.
            features (scipy.sparse matrix): Node features matrix.
            labels (np.ndarray): Node labels (class indices).
            idx_train (list): Training node indices.
            idx_val (list): Validation node indices.
            idx_test (list): Test node indices.
    """
    data_path = os.path.abspath(data_path)  # Ensures correct path format

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for name in names:
        file_path = os.path.join(data_path, f"ind.{dataset_str}.{name}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing file: {file_path}")
        
        with open(file_path, 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f)) 

    x, y, tx, ty, allx, ally, graph = objects
    test_idx_reorder = parse_index_file(os.path.join(data_path, f"ind.{dataset_str}.test.index"))
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, axis=1)

 
    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = list(range(len(y), len(y) + 500))

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    if not use_feats:
        features = sp.eye(adj.shape[0])
        
    return adj, features, labels, idx_train, idx_val, idx_test

def parse_index_file(filename):
    """
    Parse a file containing node indices (one per line).

    Args:
        filename (str): Path to index file.

    Returns:
        list: List of integer node indices.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Missing index file: {filename}")
    
    with open(filename, 'r') as f:
        return [int(line.strip()) for line in f]
