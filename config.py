import argparse
import torch
import os

parser = argparse.ArgumentParser(description='AdiHS')

parser.add_argument('--dataset', type=str, default='dblpv13', help='datasets')
parser.add_argument('--num_nodes', type=int, default=-1, help='num of nodes')
parser.add_argument('--nfeat', type=int, default=128, help='dim of input feature')
parser.add_argument('--nhid', type=int, default=32, help='dim of hidden embedding')
parser.add_argument('--nout', type=int, default=32, help='dim of output embedding')
parser.add_argument('--max_epoch', type=int, default=None, help='number of epochs to train.')
parser.add_argument('--device', type=str, default='cpu', help='training device')
parser.add_argument('--device_id', type=str, default='0', help='device id for gpu')
parser.add_argument('--seed', type=int, default=1024, help='random seed')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-7, help='weight for L2 loss on basic models.')
parser.add_argument('--model', type=str, default='AdiHS', help='model name')
parser.add_argument('--manifold', type=str, default='PoincareBall', help='Hyperbolic model')
parser.add_argument('--EPS', type=float, default=1e-15, help='eps')
parser.add_argument('--bias', type=bool, default=True, help='use bias or not')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate (1 - keep probability).')
parser.add_argument('--heads', type=int, default=1, help='structural attention heads.')
parser.add_argument('--temporal_attention_layer_heads', type=int, default=1, help='temporal_attention_layer heads')
parser.add_argument('--curvature', type=float, default=1.0, help='curvature value')
parser.add_argument('--num_walks', type=int, default=200, help='Number of walks for Node2Vec')
parser.add_argument('--workers', type=int, default=2, help='Number of workers for Node2Vec')
parser.add_argument('--Time_stamps', type=int, default=None, help='Time_stamps')
args = parser.parse_args()


if int(args.device_id) >= 0 and torch.cuda.is_available():
    args.device = torch.device("cuda".format(args.device_id))
    print('using gpu:{} to train the model'.format(args.device_id))
else:
    args.device = torch.device("cpu")
    print('using cpu to train the model')

if args.temporal_attention_layer_heads > args.nfeat:
    raise ValueError(f"temporal_attention_layer_heads ({args.temporal_attention_layer_heads}) exceeds tensor dimension ({args.nfeat}).")


