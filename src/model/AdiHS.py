import os
import sys

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, kaiming_uniform
from hgcn.layers.hyplayers import HGATConv, TemporalAttentionLayer
from hgcn.manifolds import PoincareBall

class AdiHs( nn.Module):
    """
    AdiHS GNN model for dynamic citation network anomaly detection.
    Uses hyperbolic graph attention layers and temporal attention.

    Args:
        args: Namespace of model and training hyperparameters.
        time_length (int): Number of time steps.
    """
    def __init__(self, args, time_length):
        """
        Initialize AdiHS model layers and parameters.

        Args:
            args: Namespace of model and training hyperparameters.
            time_length (int): Number of time steps.
        """
        super(AdiHs, self).__init__()
        self.manifold = PoincareBall()

        self.c = 1
        self.feat = Parameter((torch.ones(args.num_nodes, args.nfeat)).to(args.device), requires_grad=True)

        self.linear = nn.Linear(args.nfeat, args.nout)
        self.hidden_initial = torch.ones(args.num_nodes, args.nout).to(args.device)
        
        self.layer1 = HGATConv(self.manifold, args.nout, args.nhid, self.c, self.c,
                                heads=args.heads, dropout=args.dropout, att_dropout=args.dropout, concat=True)
        self.layer2 = HGATConv(self.manifold, args.nhid * args.heads, args.nout, self.c, self.c,
                                heads=args.heads, dropout=args.dropout, att_dropout=args.dropout, concat=False)
        
        self.ddy_attention_layer = TemporalAttentionLayer(
            input_dim=args.nout, 
            n_heads=args.temporal_attention_layer_heads, 
            num_time_steps=time_length,  
            attn_drop=0,  # dropout
            residual=False  
            )
        self.nhid = args.nhid
        self.nout = args.nout
        self.reset_parameters()


    def reset_parameters(self):
        """
        Reset model parameters using glorot initialization.
        """
        glorot(self.feat)
        glorot(self.linear.weight)
        

    def initHyperX(self, x, c=1.0):
        """
        Initialize node features in hyperbolic space.

        Args:
            x (torch.Tensor): Node features.
            c (float): Curvature parameter.
        Returns:
            torch.Tensor: Features in hyperbolic space.
        """
        return self.toHyperX(x, c)

    def toHyperX(self, x, c=1.0):
        """
        Map features to hyperbolic space using exponential map.

        Args:
            x (torch.Tensor): Node features.
            c (float): Curvature parameter.
        Returns:
            torch.Tensor: Features in hyperbolic space.
        """
        x_tan = self.manifold.proj_tan0(x, c)
        x_hyp = self.manifold.expmap0(x_tan, c)
        x_hyp = self.manifold.proj(x_hyp, c)
        return x_hyp

    def toTangentX(self, x, c=1.0):
        """
        Map features from hyperbolic space to tangent space.

        Args:
            x (torch.Tensor): Node features in hyperbolic space.
            c (float): Curvature parameter.
        Returns:
            torch.Tensor: Features in tangent space.
        """
        x = self.manifold.proj_tan0(self.manifold.logmap0(x, c), c)
        return x


    def forward(self, edge_index, x=None, weight=None, return_logits=False):
        """
        Forward pass of the AdiHS model.

        Args:
            edge_index (torch.Tensor): Edge indices for PyG model.
            x (torch.Tensor): Node features.
            weight: Unused.
            return_logits (bool): Unused.
        Returns:
            torch.Tensor: Output node representations.
        """
        x = self.initHyperX(self.linear(x), self.c)
        x = self.manifold.proj(x, self.c)
        x = self.layer1(x, edge_index)
        x = self.manifold.proj(x, self.c)
        x = self.layer2(x, edge_index)
        x = self.toTangentX(x, self.c) 
        return x