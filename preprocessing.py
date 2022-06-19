import numpy as np
import scipy.sparse as sp
import torch
import random
from utils import preprocess_citation_feat, normalize

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(adj_normalized, adj, features, placeholders):
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj_normalized})
    feed_dict.update({placeholders['adj_orig']: adj})
    return feed_dict

def mask_test_edges_gclWeight(adj, num_node):
    # Function to build test set with 10% positive links
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()

    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    # original_adj edges
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, val_edge_idx, axis=0)
    adj_train = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    adj = adj.multiply(adj_train)

    assert len(adj.data) == len(adj_train.data)
    sample_size = int((len(train_edges) * 1.0) / num_node)
    sample_size = sample_size + 10

    neg_list = []
    all_candiate = set(range(adj.shape[0]))
    for i in range(0, num_node):
        non_zeros = adj[i].nonzero()[1]
        neg_candi = np.array(list(all_candiate.difference(set(non_zeros))))
        if len(neg_candi) >= sample_size:
            neg_candi = np.random.choice(neg_candi, size=sample_size, replace=False)
        elif len(neg_candi) == 0:
            pass
        else:
            neg_candi = neg_candi

        neg_candi = [[i, j] for j in neg_candi]
        neg_list.extend(neg_candi)

    train_edges_false = np.array(random.sample(neg_list, len(train_edges)))
    val_edges_false = np.array(random.sample(neg_list, len(val_edges)))
    test_edges = []
    test_edges_false = []
    return adj, [train_edges, np.array(train_edges_false)], val_edges, val_edges_false, test_edges, test_edges_false

def mask_test_edges_net_lp(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    num_node = adj.shape[0]
    sample_size = int((len(train_edges) * 1.0) / num_node)
    sample_size = sample_size + 10

    neg_list = []
    all_candiate = set(range(adj.shape[0]))
    for i in range(0, num_node):
        non_zeros = adj[i].nonzero()[1]
        neg_candi = np.array(list(all_candiate.difference(set(non_zeros))))
        if len(neg_candi) >= sample_size:
            neg_candi = np.random.choice(neg_candi, size=sample_size, replace=False)
        elif len(neg_candi) == 0:
            pass
        else:
            neg_candi = neg_candi

        neg_candi = [[i, j] for j in neg_candi]
        # print('len_: %d' % len(neg_candi))
        neg_list.extend(neg_candi)
    # neg_list = np.array(neg_list)

    train_edges_false = np.array(random.sample(neg_list, len(train_edges)))

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    # assert ~ismember(test_edges_false, edges_all)
    # assert ~ismember(val_edges_false, edges_all)
    # assert ~ismember(val_edges, train_edges)
    # assert ~ismember(test_edges, train_edges)
    # assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false


def mask_test_edges_net(adj):
     # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    num_node = adj.shape[0]
    sample_size = int((len(train_edges) * 1.0) / num_node)
    sample_size = sample_size + 10

    neg_list = []
    all_candiate = set(range(adj.shape[0]))
    for i in range(0, num_node):
        non_zeros = adj[i].nonzero()[1]
        neg_candi = np.array(list(all_candiate.difference(set(non_zeros))))
        if len(neg_candi) >= sample_size:
            neg_candi = np.random.choice(neg_candi, size=sample_size, replace=False)
        elif len(neg_candi) == 0:
            pass
        else:
            neg_candi = neg_candi

        neg_candi = [[i, j] for j in neg_candi]
        neg_list.extend(neg_candi)

    train_edges_false = np.array(random.sample(neg_list, len(train_edges)))

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false


def mask_test_edges_bipartall(adj, num_node):
    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()

    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    # original_adj edges
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, val_edge_idx, axis=0)

    sample_size = int((len(train_edges) * 1.0) / num_node)
    sample_size = sample_size + 10

    neg_list = []
    all_candiate = set(range(num_node, adj.shape[0]))
    for i in range(0, num_node):
        non_zeros = adj[i].nonzero()[1]
        neg_candi = np.array(list(all_candiate.difference(set(non_zeros))))
        if len(neg_candi) >= sample_size:
            neg_candi = np.random.choice(neg_candi, size=sample_size, replace=False)
        elif len(neg_candi) == 0:
            pass
        else:
            neg_candi = neg_candi

        neg_candi = [[i, j] for j in neg_candi]
        # print('len_: %d' % len(neg_candi))
        neg_list.extend(neg_candi)

    train_edges_false = np.array(random.sample(neg_list, len(train_edges)))
    val_edges_false = np.array(random.sample(neg_list, len(val_edges)))

    # data = np.ones(train_edges.shape[0])
    data = np.array([adj[i,j] for (i, j) in train_edges])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    test_edges = []
    test_edges_false = []

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, [train_edges, np.array(train_edges_false)], val_edges, val_edges_false, test_edges, test_edges_false
