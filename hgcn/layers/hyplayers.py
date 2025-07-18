"""Hyperbolic layers."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, softmax, add_self_loops
from torch_scatter import scatter, scatter_add
from torch_geometric.nn.conv import MessagePassing, GATConv
from torch.nn.parameter import Parameter
from torch_geometric.nn.inits import glorot, zeros
from hgcn.manifolds import PoincareBall
import itertools


class HGATConv(nn.Module):
    """
    Hyperbolic graph attention convolution layer for GNNs.
    """
    def __init__(self, manifold, in_features, out_features, c_in, c_out, act=F.leaky_relu,
                 dropout=0.6, att_dropout=0.6, use_bias=True, heads=2, concat=False):
        """
        Initialize HGATConv layer.

        Args:
            manifold: Manifold object.
            in_features (int): Input feature dimension.
            out_features (int): Output feature dimension.
            c_in, c_out (float): Input/output curvature.
            act: Activation function.
            dropout, att_dropout (float): Dropout rates.
            use_bias (bool): Whether to use bias.
            heads (int): Number of attention heads.
            concat (bool): Whether to concatenate heads.
        """
        super(HGATConv, self).__init__()
        out_features = out_features * heads
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout=dropout, use_bias=use_bias)
        self.agg = HypAttAgg(manifold, c_in, out_features, att_dropout, heads=heads, concat=concat)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)
        self.manifold = manifold
        self.c_in = c_in

    def forward(self, x, edge_index):
        """
        Forward pass for HGATConv layer.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indices.
        Returns:
            torch.Tensor: Output node features.
        """
        h = self.linear.forward(x)
        h = self.agg.forward(h, edge_index)
        h = self.hyp_act.forward(h)
        return h


class HGCNConv(nn.Module):
    """
    Hyperbolic graph convolution layer (from HGCN).
    """
    def __init__(self, manifold, in_features, out_features, c_in=1.0, c_out=1.0, dropout=0.6, act=F.leaky_relu,
                 use_bias=True):
        """
        Initialize HGCNConv layer.

        Args:
            manifold: Manifold object.
            in_features (int): Input feature dimension.
            out_features (int): Output feature dimension.
            c_in, c_out (float): Input/output curvature.
            dropout (float): Dropout rate.
            act: Activation function.
            use_bias (bool): Whether to use bias.
        """
        super(HGCNConv, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout=dropout)
        self.agg = HypAgg(manifold, c_in, out_features, bias=use_bias)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)
        self.manifold = manifold
        self.c_in = c_in

    def forward(self, x, edge_index):
        """
        Forward pass for HGCNConv layer.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indices.
        Returns:
            torch.Tensor: Output node features.
        """
        h = self.linear.forward(x)
        h = self.agg.forward(h, edge_index)
        h = self.hyp_act.forward(h)
        return h


class HypLinear(nn.Module):
    """
    Hyperbolic linear transformation layer.
    """
    def __init__(self, manifold, in_features, out_features, c, dropout=0.6, use_bias=True):
        """
        Initialize HypLinear layer.

        Args:
            manifold: Manifold object.
            in_features (int): Input feature dimension.
            out_features (int): Output feature dimension.
            c (float): Curvature.
            dropout (float): Dropout rate.
            use_bias (bool): Whether to use bias.
        """
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features), requires_grad=True)
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset layer parameters using glorot and zeros initialization.
        """
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x):
        """
        Forward pass for HypLinear layer.

        Args:
            x (torch.Tensor): Node features.
        Returns:
            torch.Tensor: Output node features.
        """
        drop_weight = F.dropout(self.weight, p=self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        """
        String representation of HypLinear layer.
        """
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )


class HypAct(Module):
    """
    Hyperbolic activation layer.
    """
    def __init__(self, manifold, c_in, c_out, act):
        """
        Initialize HypAct layer.

        Args:
            manifold: Manifold object.
            c_in, c_out (float): Input/output curvature.
            act: Activation function.
        """
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def forward(self, x):
        """
        Forward pass for HypAct layer.

        Args:
            x (torch.Tensor): Node features.
        Returns:
            torch.Tensor: Activated node features.
        """
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        """
        String representation of HypAct layer.
        """
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )


class HypAggAtt(MessagePassing):
    """
    Hyperbolic aggregation layer using attention.
    """
    def __init__(self, manifold, c, out_features, bias=True):
        """
        Initialize HypAggAtt layer.

        Args:
            manifold: Manifold object.
            c (float): Curvature.
            out_features (int): Output feature dimension.
            bias (bool): Whether to use bias.
        """
        super(HypAggAtt, self).__init__()
        self.manifold = manifold
        self.c = c
        self.use_bias = bias
        self.mlp = nn.Sequential(nn.Linear(out_features * 2, 1))

    def forward(self, x, edge_index=None):
        """
        Forward pass for HypAggAtt layer.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indices.
        Returns:
            torch.Tensor: Aggregated node features.
        """
        x_tangent = self.manifold.logmap0(x, c=self.c)

        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index,
                                       num_nodes=x[1].size(self.node_dim))

        edge_i = edge_index[0]
        edge_j = edge_index[1]
        x_j = torch.nn.functional.embedding(edge_j, x_tangent)
        x_i = torch.nn.functional.embedding(edge_i, x_tangent)

        norm = self.mlp(torch.cat([x_i, x_j], dim=1))
        norm = softmax(norm, edge_i, x_i.size(0)).view(-1, 1)
        support = norm.view(-1, 1) * x_j
        support_t_curv = scatter(support, edge_i, dim=0, dim_size=x.size(0))  # aggregate the neighbors of node_i
        output = self.manifold.proj(self.manifold.expmap0(support_t_curv, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        """
        String representation of HypAggAtt layer.
        """
        return 'c={}'.format(self.c)


class HypAgg(MessagePassing):
    """
    Hyperbolic aggregation layer using degree.
    """
    def __init__(self, manifold, c, out_features, bias=True):
        """
        Initialize HypAgg layer.

        Args:
            manifold: Manifold object.
            c (float): Curvature.
            out_features (int): Output feature dimension.
            bias (bool): Whether to use bias.
        """
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.manifold = PoincareBall()
        self.c = c
        self.use_bias = bias
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        zeros(self.bias)
        self.mlp = nn.Sequential(nn.Linear(out_features * 2, 1))

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
        """
        Compute normalized edge weights for aggregation.

        Args:
            edge_index (torch.Tensor): Edge indices.
            num_nodes (int): Number of nodes.
            edge_weight (torch.Tensor, optional): Edge weights.
            improved (bool): Use improved normalization.
            dtype: Data type.
        Returns:
            tuple: (edge_index, normalized edge weights)
        """
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index=None):
        """
        Forward pass for HypAgg layer.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indices.
        Returns:
            torch.Tensor: Aggregated node features.
        """
        x_tangent = self.manifold.logmap0(x, c=self.c)
        edge_index, norm = self.norm(edge_index, x.size(0), dtype=x.dtype)
        node_i = edge_index[0]
        node_j = edge_index[1]
        x_j = torch.nn.functional.embedding(node_j, x_tangent)
        support = norm.view(-1, 1) * x_j
        support_t = scatter(support, node_i, dim=0, dim_size=x.size(0))  # aggregate the neighbors of node_i
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        """
        String representation of HypAgg layer.
        """
        return 'c={}'.format(self.c)


class HypAttAgg(MessagePassing):
    """
    Hyperbolic attention aggregation layer.
    """
    def __init__(self, manifold, c, out_features, att_dropout=0.6, heads=1, concat=False):
        """
        Initialize HypAttAgg layer.

        Args:
            manifold: Manifold object.
            c (float): Curvature.
            out_features (int): Output feature dimension.
            att_dropout (float): Attention dropout rate.
            heads (int): Number of attention heads.
            concat (bool): Whether to concatenate heads.
        """
        super(HypAttAgg, self).__init__()
        self.manifold = manifold
        self.dropout = att_dropout
        self.out_channels = out_features // heads
        self.negative_slope = 0.2
        self.heads = heads
        self.c = c
        self.concat = concat
        self.att_i = Parameter(torch.Tensor(1, heads, self.out_channels), requires_grad=True)
        self.att_j = Parameter(torch.Tensor(1, heads, self.out_channels), requires_grad=True)
        glorot(self.att_i)
        glorot(self.att_j)

    def forward(self, x, edge_index):
        """
        Forward pass for HypAttAgg layer.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge indices.
        Returns:
            torch.Tensor: Aggregated node features.
        """
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index,
                                       num_nodes=x.size(self.node_dim))

        edge_index_i = edge_index[0]
        edge_index_j = edge_index[1]

        x_tangent0 = self.manifold.logmap0(x, c=self.c)  # project to origin
        x_i = torch.nn.functional.embedding(edge_index_i, x_tangent0)
        x_j = torch.nn.functional.embedding(edge_index_j, x_tangent0)
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)

        alpha = (x_i * self.att_i).sum(-1) + (x_j * self.att_j).sum(-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes=x_i.size(0))
        alpha = F.dropout(alpha, self.dropout, training=self.training)
        support_t = scatter(x_j * alpha.view(-1, self.heads, 1), edge_index_i, dim=0)

        if self.concat:
            support_t = support_t.view(-1, self.heads * self.out_channels)
        else:
            support_t = support_t.mean(dim=1)
        support_t = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)

        return support_t

# refer to: https://github.com/ferrine/hyrnn/blob/master/hyrnn/nets.py
class HypGRU(nn.Module):
    """
    Hyperbolic GRU recurrent layer for dynamic graphs.
    """
    def __init__(self, args):
        """
        Initialize HypGRU layer.

        Args:
            args: Namespace of model hyperparameters.
        """
        super(HypGRU, self).__init__()
        self.manifold = PoincareBall()
        self.nhid = args.nhid
        self.weight_ih = Parameter(torch.Tensor(3 * args.nhid, args.nhid), requires_grad=True).to(args.device)
        self.weight_hh = Parameter(torch.Tensor(3 * args.nhid, args.nhid), requires_grad=True).to(args.device)
        if args.bias:
            bias = nn.Parameter(torch.zeros(3, args.nhid) * 1e-5, requires_grad=False)
            self.bias = self.manifold.expmap0(bias).to(args.device)
        else:
            self.register_buffer("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset GRU parameters using uniform initialization.
        """
        stdv = 1.0 / math.sqrt(self.nhid)
        for weight in itertools.chain.from_iterable([self.weight_ih, self.weight_hh]):
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, hyperx, hyperh):
        """
        Forward pass for HypGRU layer.

        Args:
            hyperx (torch.Tensor): Input features.
            hyperh (torch.Tensor): Hidden state.
        Returns:
            torch.Tensor: Output hidden state.
        """
        out = self.mobius_gru_cell(hyperx, hyperh, self.weight_ih, self.weight_hh, self.bias)
        return out

    def mobius_gru_cell(self, input, hx, weight_ih, weight_hh, bias, nonlin=None, ):
        """
        Mobius GRU cell computation in hyperbolic space.

        Args:
            input (torch.Tensor): Input features.
            hx (torch.Tensor): Hidden state.
            weight_ih, weight_hh (torch.Tensor): Input/hidden weights.
            bias (torch.Tensor): Bias terms.
            nonlin: Nonlinearity function (optional).
        Returns:
            torch.Tensor: Output hidden state.
        """
        W_ir, W_ih, W_iz = weight_ih.chunk(3)
        b_r, b_h, b_z = bias
        W_hr, W_hh, W_hz = weight_hh.chunk(3)

        z_t = self.manifold.logmap0(self.one_rnn_transform(W_hz, hx, W_iz, input, b_z)).sigmoid()
        r_t = self.manifold.logmap0(self.one_rnn_transform(W_hr, hx, W_ir, input, b_r)).sigmoid()

        rh_t = self.manifold.mobius_pointwise_mul(r_t, hx)
        h_tilde = self.one_rnn_transform(W_hh, rh_t, W_ih, input, b_h)

        if nonlin is not None:
            h_tilde = self.manifold.mobius_fn_apply(nonlin, h_tilde)
        delta_h = self.manifold.mobius_add(-hx, h_tilde)
        h_out = self.manifold.mobius_add(hx, self.manifold.mobius_pointwise_mul(z_t, delta_h))
        return h_out

    def one_rnn_transform(self, W, h, U, x, b):
        """
        Mobius linear transformation for GRU cell.

        Args:
            W, U (torch.Tensor): Weight matrices.
            h, x (torch.Tensor): Hidden/input features.
            b (torch.Tensor): Bias.
        Returns:
            torch.Tensor: Transformed features.
        """
        W_otimes_h = self.manifold.mobius_matvec(W, h)
        U_otimes_x = self.manifold.mobius_matvec(U, x)
        Wh_plus_Ux = self.manifold.mobius_add(W_otimes_h, U_otimes_x)
        return self.manifold.mobius_add(Wh_plus_Ux, b)

    def mobius_linear(self, input, weight, bias=None, hyperbolic_input=True, hyperbolic_bias=True, nonlin=None):
        """
        Mobius linear transformation in hyperbolic space.

        Args:
            input (torch.Tensor): Input features.
            weight (torch.Tensor): Weight matrix.
            bias (torch.Tensor, optional): Bias.
            hyperbolic_input (bool): Whether input is hyperbolic.
            hyperbolic_bias (bool): Whether bias is hyperbolic.
            nonlin: Nonlinearity function (optional).
        Returns:
            torch.Tensor: Transformed features.
        """
        if hyperbolic_input:
            output = self.manifold.mobius_matvec(weight, input)
        else:
            output = torch.nn.functional.linear(input, weight)
            output = self.manifold.expmap0(output)
        if bias is not None:
            if not hyperbolic_bias:
                bias = self.manifold.expmap0(bias)
            output = self.manifold.mobius_add(output, bias)
        if nonlin is not None:
            output = self.manifold.mobius_fn_apply(nonlin, output)
        output = self.manifold.project(output)
        return output


class TemporalAttentionLayer(nn.Module):
    """
    Temporal attention layer for dynamic graph neural networks.
    """
    def __init__(self, 
                input_dim, 
                n_heads, 
                num_time_steps, 
                attn_drop, 
                residual):
        """
        Initialize TemporalAttentionLayer.

        Args:
            input_dim (int): Input feature dimension.
            n_heads (int): Number of attention heads.
            num_time_steps (int): Number of time steps.
            attn_drop (float): Attention dropout rate.
            residual (bool): Whether to use residual connections.
        """
        super(TemporalAttentionLayer, self).__init__()
        self.n_heads = n_heads
        self.num_time_steps = num_time_steps
        self.residual = residual

        # define weights
        self.position_embeddings = nn.Parameter(torch.Tensor(num_time_steps, input_dim))  # 位置embedding信息[16, 128]
        self.Q_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))  # [128, 128]; W*Q
        self.K_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.V_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        # ff
        self.lin = nn.Linear(input_dim, input_dim, bias=True)
        # dropout 
        self.attn_dp = nn.Dropout(attn_drop)
        self.xavier_init()

    def forward(self, inputs):
        """In:  attn_outputs (of StructuralAttentionLayer at each snapshot):= [N, T, F]"""
        # 1: Add position embeddings to input; [143, 16]: 143个节点，每个节点16个位置信息
        position_inputs = torch.arange(0,self.num_time_steps).reshape(1, -1).repeat(inputs.shape[0], 1).long().to(inputs.device)  # 重复143个节点; 每个节点有16个时间步
        temporal_inputs = inputs + self.position_embeddings[position_inputs] # [N, T, F]; 每个节点在各个时刻对应到的128维向量

        # 2: Query, Key based multi-head self attention. [143, 16, 128]
        q = torch.tensordot(temporal_inputs, self.Q_embedding_weights, dims=([2],[0])) # [N, T, F]; 第一个矩阵第2个维度，乘以，第二个矩阵的第0个维度
        k = torch.tensordot(temporal_inputs, self.K_embedding_weights, dims=([2],[0])) # [N, T, F]
        v = torch.tensordot(temporal_inputs, self.V_embedding_weights, dims=([2],[0])) # [N, T, F]

        # 3: Split, concat and scale.
        # Ensure split_size is valid
        if self.n_heads > q.shape[-1]:
            raise ValueError(f"Number of heads ({self.n_heads}) exceeds tensor dimension ({q.shape[-1]}).")
        split_size = max(1, int(q.shape[-1]/self.n_heads))  # Ensure split_size is at least 1
        q_ = torch.cat(torch.split(q, split_size_or_sections=split_size, dim=2), dim=0) # [hN, T, F/h]
        k_ = torch.cat(torch.split(k, split_size_or_sections=split_size, dim=2), dim=0) # [hN, T, F/h]
        v_ = torch.cat(torch.split(v, split_size_or_sections=split_size, dim=2), dim=0) # [hN, T, F/h]
        
        outputs = torch.matmul(q_, k_.permute(0,2,1)) # [hN, T, T]
        outputs = outputs / (self.num_time_steps ** 0.5)  # Q*K
        # 4: Masked (causal) softmax to compute attention weights. 目的是将之前没有出现的时间步，设置为0;
        diag_val = torch.ones_like(outputs[0])  # [16,16]的全1向量
        tril = torch.tril(diag_val)  # 下三角阵
        masks = tril[None, :, :].repeat(outputs.shape[0], 1, 1) # [h*N, T, T]  重复N次（2288）; [2288, 16, 16]
        padding = torch.ones_like(masks) * (-2**32+1)  # 负无穷
        outputs = torch.where(masks==0, padding, outputs)  # outputs中mask为0的地方，填充padding中负无穷的数值
        outputs = F.softmax(outputs, dim=2)  # output:[2288, 16, 16]
        self.attn_wts_all = outputs # [h*N, T, T]
                
        # 5: Dropout on attention weights.
        # if self.training:
        #     outputs = self.attn_dp(outputs)  # dropout
        outputs = torch.matmul(outputs, v_)  # [hN, T, F/h]  # (K*Q)*V; ouput-经过归一化后的attention系数[2288, 16, 16]
        outputs = torch.cat(torch.split(outputs, split_size_or_sections=int(outputs.shape[0]/self.n_heads), dim=0), dim=2) # [N, T, F]
        
        # 6: Feedforward and residual
        #outputs = self.feedforward(outputs)
        # if self.residual:
        #     outputs = outputs + temporal_inputs
        return outputs  # 所有节点聚合时序self-attention后的节点embedding，所有时间

    def feedforward(self, inputs):
        """
        Feedforward layer for temporal attention.

        Args:
            inputs (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor after feedforward and residual.
        """
        outputs = F.relu(self.lin(inputs))
        return outputs + inputs


    def xavier_init(self):
        """
        Xavier uniform initialization for all learnable parameters.
        """
        nn.init.xavier_uniform_(self.position_embeddings)
        nn.init.xavier_uniform_(self.Q_embedding_weights)
        nn.init.xavier_uniform_(self.K_embedding_weights)
        nn.init.xavier_uniform_(self.V_embedding_weights)