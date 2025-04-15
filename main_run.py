import os
import sys
import time
from collections import OrderedDict
import numpy as np
import torch
import torch.optim as optim
import networkx as nx
from node2vec import Node2Vec
from model.AdiHS import AdiHs
from config import args
from utilis import data_utilis
from torch_geometric.utils import from_scipy_sparse_matrix



"""BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)"""

class Runner(object):
    def __init__(self, data):
        self.data = data
        self.len = data['time_length']
        self.start_train = 0
        self.train_shots = list(range(0, self.len - args.testlength))
        self.test_shots = list(range(self.len - args.testlength, self.len))

        # Node2Vec Embeddings
        adj = self.data['adj']
        G = nx.from_scipy_sparse_array(adj)
        node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)
        model_n2v = node2vec.fit(window=10, min_count=1, batch_words=4)
        embeddings = np.array([model_n2v.wv[str(node)] for node in G.nodes()])
        self.embeddings = torch.FloatTensor(embeddings).to(args.device)

        args.nfeat = self.embeddings.size(1)
        self.x = self.embeddings
        self.model = AdiHs(args, len(self.train_shots)).to(args.device)

    def run(self, run_manager):
        optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        t_total0 = time.time()
        test_results, min_loss = [0] * 5, 100000
        best_epoch_auc = 0
        patient_auc = 0

        self.model.train()
        for epoch in range(1, args.max_epoch + 1):
            run_manager.begin_epoch()
            t0 = time.time()

            structural_out = []
            for t in self.train_shots:
                edge_index, pos_index, neg_index, activate_nodes, edge_weight, _, _ = prepare(self.data, t)
                z = self.model(edge_index, self.x)  # [num_nodes, nout]
                structural_out.append(z)

            structural_outputs = [g[:, None, :] for g in structural_out]
            maximum_node_num = structural_outputs[-1].shape[0]
            out_dim = structural_outputs[-1].shape[-1]
            structural_outputs_padded = []
            for out in structural_outputs:
                zero_padding = torch.zeros(maximum_node_num - out.shape[0], 1, out_dim).to(out.device)
                padded = torch.cat((out, zero_padding), dim=0)
                structural_outputs_padded.append(padded)

            structural_outputs_padded = torch.cat(structural_outputs_padded, dim=1)  # [N, T, F]
            temporal_out = self.model.ddy_attention_layer(structural_outputs_padded)
    """here will be the anomaly detection part"""
          
if __name__ == '__main__':
    params = OrderedDict(
        nhid=[args.nhid],
        nout=[args.nout],
        temporal_attention_layer_heads=[args.temporal_attention_layer_heads],
        heads=[args.heads],
        dataset=[args.dataset],
        split_count=[args.split_count]
    )

    run_manager = RunManager()
    for run in RunBuilder.get_runs(params):
        run_manager.begin_run(run)

        args.nhid = run.nhid
        args.nout = run.nout
        args.temporal_attention_layer_heads = run.temporal_attention_layer_heads
        args.heads = run.heads
        args.dataset = run.dataset
        args.split_count = f'{run.split_count}-split'
        args.output_folder = f'../data/output/log/{args.dataset}/{args.model}/{args.split_count}/'

        data = loader(dataset=args.dataset, split_count=args.split_count)
        args.num_nodes = data['num_nodes']
        set_random(args.seed)
        init_logger(prepare_dir(args.output_folder) + args.dataset + '.txt')

        try:
            runner = Runner(data)
            runner.run(run_manager)
            run_manager.save(args.output_folder, args.split_count, args.heads, args.temporal_attention_layer_heads, args.nout, args.nhid)
        except RuntimeError:
            logger.info(f'current args except runtime error')

        logger.info(f'current args: {args}')