import numpy as np
import scipy.sparse as sp
import torch
import pickle as pkl
from normalization import fetch_normalization, row_normalize
import sys
import networkx as nx
import os
import random
import scipy.io as scio


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape=loader['shape'])


def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    print('Finish loading dataset')

    return adj, features, labels, idx_train, idx_val, idx_test


def preprocess_citation(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    features = row_normalize(features)
    return adj, features

def preprocess_citation_feat(features):
    features = row_normalize(features)
    return features


def preprocess_citation_graph(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    adj2 = features * features.T
    adj2 = adj_normalizer(adj2)
    features = row_normalize(features)
    return adj, features, adj2


def preprocess_citation_bigraph(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    adj_cn = features.T
    features = row_normalize(features)
    adj_cn = row_normalize(adj_cn)
    adj_nc = features
    return adj, features, adj_nc, adj_cn


def load_citationCONN_continuous(dataset_str="BlogCatalog", knn=0, use_net=1, ratio=0.0):
    data_file = 'data/{}/{}'.format(dataset_str, dataset_str) + '.mat'
    data = scio.loadmat(data_file)
    if dataset_str == 'ACM':
        features = data['Features']
    else:
        features = data['Attributes']
    labels = data['Label'].reshape(-1)
    adj = data['Network']

    label_min = np.min(labels)
    if label_min != 0:
        labels = labels - 1
    max_class = np.max(labels) + 1
    class_one = np.eye(max_class)
    labels = class_one[labels]

    if type(features) is not np.ndarray:
        features = features.toarray()
        features[features < 0] = 0.
    if use_net:
        pass
    else:
        if knn == 0:
            if ratio == 0:
                ratio = np.median(features)
            features[features <= ratio] = 0.
        else:
            # feat_index = (-features).argsort()[:knn]
            feat_index = features.argsort()
            feat_index = feat_index[:, -knn:]
            feat_value = np.array([features[i][index] for i, index in enumerate(feat_index)])
            features = np.zeros_like(features)
            for i in range(features.shape[0]):
                indx = feat_index[i]
                features[i, indx] = feat_value[i]
            pass
    features = sp.csr_matrix(features)
    print('---> Total edges for attributes is %d' % features.nnz)

    n, d = features.shape
    sparse_mx = features.tocoo().astype(np.float32)
    index_row = sparse_mx.row
    index_col = sparse_mx.col
    nodeAttriNet = sp.csr_matrix((sparse_mx.data, (index_row, index_col)), shape=(n, d))
    adj_mat = sp.dok_matrix((n + d, n + d), dtype=np.float32)
    adj_mat = adj_mat.tolil()
    R = nodeAttriNet.tolil()
    adj_mat[:n, n:] = R
    adj_mat[n:, :n] = R.T

    adj.data = adj.data
    adj_ori = adj.tolil()
    adj_mat[:n, :n] = adj_ori

    adj_mat = adj_mat.tocsr()
    sum_num = adj_mat.nnz * 1.0
    sparsity = sum_num / (n * d)
    print('---> Sparsity of feature matrix: %.4f' % sparsity)

    idx_train = np.array(range(500))
    idx_val = np.array(range(500, 1000))
    idx_test = np.array(range(1000, 1500))

    return adj, features, adj_mat, labels, idx_train, idx_val, idx_test


def load_citationANEmatWeight(dataset_str="BlogCatalog"):
    data_file = 'data/{}/{}'.format(dataset_str, dataset_str) + '.mat'
    data = scio.loadmat(data_file)
    if dataset_str == 'ACM':
        features = data['Features']
    else:
        features = data['Attributes']
    labels = data['Label'].reshape(-1)
    adj = data['Network']

    label_min = np.min(labels)
    if label_min != 0:
        labels = labels - 1
    max_class = np.max(labels) + 1
    class_one = np.eye(max_class)
    labels = class_one[labels]

    n, d = features.shape
    sparse_mx = features.tocoo().astype(np.float32)
    index_row = sparse_mx.row
    index_col = sparse_mx.col
    nodeAttriNet = sp.csr_matrix((sparse_mx.data, (index_row, index_col)), shape=(n, d))
    adj_mat = sp.dok_matrix((n + d, n + d), dtype=np.float32)
    adj_mat = adj_mat.tolil()
    R = nodeAttriNet.tolil()
    adj_mat[:n, n:] = R
    adj_mat[n:, :n] = R.T
    adj.data = adj.data
    adj_ori = adj.tolil()
    adj_mat[:n, :n] = adj_ori
    adj_mat = adj_mat.tocsr()
    sum_num = adj_mat.nnz * 1.0
    sparsity = sum_num / (n * d)
    print('---> Sparsity of feature matrix: %.4f' % sparsity)

    idx_train = np.array(range(500))
    idx_val = np.array(range(500, 1000))
    idx_test = np.array(range(1000, 1500))

    return adj, features, adj_mat, labels, idx_train, idx_val, idx_test

def load_citationANEWeight(dataset_str="cora"):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended
        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)

        # adj, features = preprocess_citation(adj, features, normalization)
        # features = preprocess_citation_feat(features)

    elif dataset_str == 'nell.0.001':
        # Find relation nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(allx.shape[0], len(graph))
        isolated_node_idx = np.setdiff1d(test_idx_range_full, test_idx_reorder)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - allx.shape[0], :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - allx.shape[0], :] = ty
        ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]

        idx_all = np.setdiff1d(range(len(graph)), isolated_node_idx)

        if not os.path.isfile("data/{}.features.npz".format(dataset_str)):
            print("Creating feature vectors for relations - this might take a while...")
            features_extended = sp.hstack((features, sp.lil_matrix((features.shape[0], len(isolated_node_idx)))),
                                          dtype=np.int32).todense()
            features_extended[isolated_node_idx, features.shape[1]:] = np.eye(len(isolated_node_idx))
            features = sp.csr_matrix(features_extended)
            print("Done!")
            save_sparse_csr("data/{}.features".format(dataset_str), features)
        else:
            features = load_sparse_csr("data/{}.features.npz".format(dataset_str))
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        # adj = adj.astype(float)
        features = features.astype(float)
        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)

    else:
        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)

    n, d = features.shape
    sparse_mx = features.tocoo().astype(np.float32)
    index_row = sparse_mx.row
    index_col = sparse_mx.col
    nodeAttriNet = sp.csr_matrix((sparse_mx.data, (index_row, index_col)), shape=(n, d))
    adj_mat = sp.dok_matrix((n + d, n + d), dtype=np.float32)
    adj_mat = adj_mat.tolil()
    R = nodeAttriNet.tolil()
    adj_mat[:n, n:] = R
    adj_mat[n:, :n] = R.T
    adj.data = adj.data
    adj_ori = adj.tolil()
    adj_mat[:n, :n] = adj_ori
    adj_mat = adj_mat.tocsr()
    sum_num = adj_mat.nnz * 1.0
    sparsity = sum_num / (n * d)
    print('---> Sparsity of feature matrix: %.4f' % sparsity)

    return adj, features, adj_mat, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))

    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_array(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.sum(mx, axis=1)

    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    # r_mat_inv = sp.diags(r_inv)
    # r_mat_inv = np.diag(r_inv)
    for i, weight_ in enumerate(r_inv):
        row = mx[i] * weight_
        mx[i] = row
    # mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def create_sparse_eye_tensor(shape):
    row = np.array(range(shape[0])).astype(np.int64)
    col = np.array(range(shape[0])).astype(np.int64)
    value_ = np.ones(shape[0]).astype(float)
    indices = torch.from_numpy(np.vstack((row, col)))
    values = torch.from_numpy(value_)
    shape = torch.Size(shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class EdgeSampler(object):
    def __init__(self, train_edges, train_edge_false, batch_size, remain_delet=True, shuffle=True):
        self.shuffle = shuffle
        self.index = 0
        self.index_false = 0
        self.pos_edge = train_edges
        self.neg_edge = train_edge_false
        self.id_index = list(range(train_edges.shape[0]))
        self.data_len = len(self.id_index)
        self.remain_delet = remain_delet
        self.batch_size = batch_size
        if self.shuffle:
            self._shuffle()

    def __iter__(self):
        return self

    def _shuffle(self):
        random.shuffle(self.id_index)

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.remain_delet:
            if self.index + self.batch_size > self.data_len:
                self.index = 0
                self.index_false = 0
                self._shuffle()
                raise StopIteration
            batch_index = self.id_index[self.index: self.index + self.batch_size]
            batch_x = self.pos_edge[batch_index]
            batch_y = self.neg_edge[batch_index]
            self.index += self.batch_size

        else:
            if self.index >= self.data_len:
                self.index = 0
                raise StopIteration
            end_ = min(self.index + self.batch_size, self.data_len)
            batch_index = self.id_index[self.index: end_]
            batch_x = self.pos_edge[batch_index]
            batch_y = self.neg_edge[batch_index]
            self.index += self.batch_size
        return np.array(batch_x), np.array(batch_y)

def load_citation(dataset_str="cora"):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended
        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)

        # adj, features = preprocess_citation(adj, features, normalization)
        features = preprocess_citation_feat(features)

    elif dataset_str == 'nell.0.001':
        # Find relation nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(allx.shape[0], len(graph))
        isolated_node_idx = np.setdiff1d(test_idx_range_full, test_idx_reorder)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - allx.shape[0], :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - allx.shape[0], :] = ty
        ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]

        idx_all = np.setdiff1d(range(len(graph)), isolated_node_idx)

        if not os.path.isfile("data/{}.features.npz".format(dataset_str)):
            print("Creating feature vectors for relations - this might take a while...")
            features_extended = sp.hstack((features, sp.lil_matrix((features.shape[0], len(isolated_node_idx)))),
                                          dtype=np.int32).todense()
            features_extended[isolated_node_idx, features.shape[1]:] = np.eye(len(isolated_node_idx))
            features = sp.csr_matrix(features_extended)
            print("Done!")
            save_sparse_csr("data/{}.features".format(dataset_str), features)
        else:
            features = load_sparse_csr("data/{}.features.npz".format(dataset_str))
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        # adj = adj.astype(float)
        features = features.astype(float)
        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)

        features = preprocess_citation_feat(features)

    else:
        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)

        # adj, features = preprocess_citation(adj, features, normalization)
        features = preprocess_citation_feat(features)

    return adj, features, labels, idx_train, idx_val, idx_test

def load_citationANEmat(dataset_str="BlogCatalog"):
    data_file = 'data/{}/{}'.format(dataset_str, dataset_str) + '.mat'
    data = scio.loadmat(data_file)
    if dataset_str == 'ACM':
        features = data['Features']
    else:
        features = data['Attributes']
    labels = data['Label'].reshape(-1)
    adj = data['Network']
    features = preprocess_citation_feat(features)

    label_min = np.min(labels)
    if label_min != 0:
        labels = labels - 1
    max_class = np.max(labels) + 1
    class_one = np.eye(max_class)
    labels = class_one[labels]

    n, d = features.shape
    sparse_mx = features.tocoo().astype(np.float32)
    index_row = sparse_mx.row
    index_col = sparse_mx.col
    nodeAttriNet = sp.csr_matrix((np.ones_like(index_row), (index_row, index_col)), shape=(n, d))
    adj_mat = sp.dok_matrix((n + d, n + d), dtype=np.float32)
    adj_mat = adj_mat.tolil()
    R = nodeAttriNet.tolil()
    adj_mat[:n, n:] = R
    adj_mat[n:, :n] = R.T
    adj_mat = adj_mat.tocsr()
    sum_num = adj_mat.sum() * 1.0
    sparsity = sum_num / (n * d)
    print('---> Sparsity of feature matrix: %.4f' % sparsity)

    idx_train = np.array(range(500))
    idx_val = np.array(range(500, 1000))
    idx_test = np.array(range(1000, 1500))

    return adj, features, adj_mat, labels, idx_train, idx_val, idx_test


if __name__ == '__main__':
    dataset = 'BlogCatalog'
    # data = load_citationmat(dataset)