import math
import numpy as np
import torch
from config import args


def uniform(size, tensor):
    """
    Initialize tensor with uniform distribution in [-1/sqrt(size), 1/sqrt(size)].

    Args:
        size (int): Size parameter for bound calculation.
        tensor (torch.Tensor): Tensor to initialize.
    Returns:
        None. Modifies tensor in-place.
    """
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


def xavier_init(shape):
    """
    Xavier (Glorot & Bengio) initialization for numpy arrays.

    Args:
        shape (tuple): Shape of the array to initialize.
    Returns:
        torch.Tensor: Initialized tensor.
    """
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = np.random.uniform(low=-init_range, high=init_range, size=shape)
    return torch.Tensor(initial)


def glorot(tensor):
    """
    Xavier (Glorot) initialization for PyTorch tensors.

    Args:
        tensor (torch.Tensor): Tensor to initialize.
    Returns:
        None. Modifies tensor in-place.
    """
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)





def prepare(data, t, detection=False):
    """
    Prepare edge indices and node lists for training or detection at time t.

    Args:
        data (dict): Data dictionary containing edge indices and node lists.
        t (int): Time step.
        detection (bool): If True, prepare for detection; else for training.
    Returns:
        tuple: If detection is False, returns
            (edge_index, pos_index, neg_index, nodes, weights, new_pos_index, new_neg_index).
        If detection is True, returns
            (train_pos_edge_index, val_pos_edge_index, val_neg_edge_index, test_pos_edge_index, test_neg_edge_index).
    """
    if detection == False:
        # obtain adj index
        edge_index = data['edge_index_list'][t].long().to(args.device)  # torch edge index
        pos_index = data['pedges'][t].long().to(args.device)  # torch edge index
        neg_index = data['nedges'][t].long().to(args.device)  # torch edge index
        new_pos_index = data['new_pedges'][t].long().to(args.device)  # torch edge index
        new_neg_index = data['new_nedges'][t].long().to(args.device)  # torch edge index
        # 2.Obtain current updated nodes
        # nodes = list(np.intersect1d(pos_index.numpy(), neg_index.numpy()))
        # 2.Obtain full related nodes
        nodes = list(np.union1d(pos_index.cpu().numpy(), neg_index.cpu().numpy()))
        weights = None
        return edge_index, pos_index, neg_index, nodes, weights, new_pos_index, new_neg_index

    if detection == True:
        train_pos_edge_index = data['gdata'][t].train_pos_edge_index.long().to(args.device)

        val_pos_edge_index = data['gdata'][t].val_pos_edge_index.long().to(args.device)
        val_neg_edge_index = data['gdata'][t].val_neg_edge_index.long().to(args.device)

        test_pos_edge_index = data['gdata'][t].test_pos_edge_index.long().to(args.device)
        test_neg_edge_index = data['gdata'][t].test_neg_edge_index.long().to(args.device)
        return train_pos_edge_index, val_pos_edge_index, val_neg_edge_index, test_pos_edge_index, test_neg_edge_index