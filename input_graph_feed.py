import numpy as np
import time
import os
import torch
import pickle
import scipy.sparse as sp
import argparse
import math

from graph import BiGraph, HomeGraph
from iterators import BatchedIterator
from sampler import IterateNodeSampler, EdgeSampler, RandomNegativeSampler, NodeSampler, \
    RandomNHopNeighborSampler_homoge, RandomNHopNeighborSampler_BI

DEFAULT_BATCH_SIZE = 32

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def interleaving_seq(a, b, length):
    """
    return a,b,a,b,a or a,b,b,b
    :param a:
    :param b:
    :param length:
    :return:
    """
    return [a, b] * (length//2) + [a] * (length % 2)


def get_edge_iterator(edges, batch_size=512, shuffle=False, epochs=-1):
    temp = []
    temp.append(edges.src)
    temp.append(edges.dest)
    edges = np.array(list(zip(*temp)))
    # edges = np.array([(e[0], e[1]) for e in edges])
    batched_iterator = BatchedIterator(edges, batch_size, shuffle=shuffle, epochs=epochs)
    return batched_iterator


def Batch_feat(feat_cat, feat_cont):
    ret = dict()
    ret['categorical'] = feat_cat
    ret['continuous'] = feat_cont

    return ret
def get_category_embedding_size(x):
    if x < 20:
        return 4
    elif x < 1000:
        return 8
    elif x < 10000:
        return 16
    elif x < 100000:
        return 32
    else:
        return 32
        # return 64


class GraphInput(object):
    """
    An Input Manager that iteratively generate input feeds for training
    It takes a bipartite graph as input,
    specifically, ui_edges, ii_edges, u_attrs, and i_attrs
    """

    def __init__(
            self, flag
    ):
        self.graph = None
        self.graph_value = None
        self.X = None
        self.Y = None
        self.num_class = None
        self.train_index = None
        self.valid_index = None
        self.test_index = None
        self.node_num = None
        self.node_dim = None
        self.node_iter = None
        self.edge_iter_train = None
        self.edge_iter_valid = None
        self.adj = None
        self.train_edge = None
        self.test_edge = None
        self.node_iter_valid = None
        self.node_iter_test = None
        self.neigbor_sampler = None
        self.neg_sampler = None
        self.batch_size = flag.batch_size
        self.dataset = flag.dataset
        self.use_feat = flag.use_feat
        self.para = flag
        self.u_node_iter = None
        self.u_neighs_num = [] if flag.layer_num == '' else [int(x) for x in flag.layer_num.split(',')]
        self.u_depth = len(self.u_neighs_num)
        self.avg_neigh = None
        self.read_data_from_edges()
        self.u_neighs_num = [self.avg_neigh + 2 for i in range(self.u_depth)]
        print('---> Layer-wise average number of neighbor is {}'.format(self.u_neighs_num))

    def generate_train_index(self):
        all_index = set(list(range(self.node_num)))
        test_index = set(list(self.test_index))
        valid_index = set(list(self.valid_index))
        train_index = all_index - test_index - valid_index
        self.train_index = np.array(list(train_index))

    def train_valid_edge_split(self, graph_):
        src, dst = graph_[0], graph_[1]
        node_dict = {}
        for i, src_ in enumerate(src):
            dest_ = dst[i]
            if src_ >= dest_:
                continue
            if src_ not in node_dict:
                node_dict.setdefault(src_, [])
            node_dict[src_].append(dest_)

        edges = []
        for key, value in node_dict.items():
            value_ = [[key, i] for i in value]
            edges.extend(value_)
        edges = np.array(edges)
        edges_idx = list(range(edges.shape[0]))
        np.random.shuffle(edges_idx)
        num_val = int(np.floor(edges.shape[0] / 20.))
        val_edge_idx = edges_idx[:num_val]
        train_edge_idx = edges_idx[num_val:]
        self.train_edge = edges[train_edge_idx].tolist()
        self.test_edge = edges[val_edge_idx].tolist()

        # for src_, dest_ in node_dict.items():
        #     np.random.shuffle(dest_)
        #     if len(dest_) >= 2:
        #         num_valid = math.ceil(len(dest_) * 0.1)
        #         train_edge = [[src_, i] for i in dest_[0:-num_valid]]
        #         test_edge = [[src_, i] for i in dest_[-num_valid:]]
        #         self.train_edge.extend(train_edge)
        #         self.test_edge.extend(test_edge)
        #     else:
        #         train_edge = [[src_, i] for i in dest_]
        #         self.train_edge.extend(train_edge)
        print('# of train_edge is %d test_edge is %d' % (len(self.train_edge), len(self.test_edge)))

    def read_data_from_edges(self):
        print('--> Loading {} dataset...'.format(self.dataset))
        data_file = 'data/' + self.para.dataset + '/{}.pkl'.format(self.para.dataset)
        with open(data_file, 'rb') as f:
            graph_, self.graph_value, self.X, self.Y, self.train_index,\
            self.valid_index, self.test_index = pickle.load(f)

        self.node_num, self.node_dim = self.X.shape
        if self.use_feat == 0:
            self.X = sp.sparse.csr_matrix(np.eye(self.node_num))
            self.node_dim = self.X.shape[1]
        self.num_class = np.max(self.Y) + 1
        self.train_valid_edge_split(graph_)
        src, dest = graph_[0], graph_[1]
        adj = sp.coo_matrix((self.graph_value, (src, dest)))
        self.adj = adj.tocsr()
        self.graph = HomeGraph(src, dest)
        self.avg_neigh = int(self.graph.avg_neigh)
        if self.para.semi == 0:
            self.generate_train_index()

        print('{} # of node {} dim {} edges {} avg_num={}'.format(self.dataset, self.node_num, self.node_dim,
                                                                 graph_.shape[1], self.graph.avg_neigh))
        print('train_set: {} valid_set: {} test_set: {}'.format(self.train_index.shape[0],
                                                                self.valid_index.shape[0], self.test_index.shape[0]))

    def init_server(self):
        batch_size = self.batch_size
        self.neigbor_sampler = RandomNHopNeighborSampler_homoge(self.u_neighs_num)
        test_index = np.array(list(range(self.node_num)))
        self.node_iter = NodeSampler(test_index, batch_size, remain_delet=False, shuffle=True)
        self.edge_iter_train = EdgeSampler(self.train_edge, batch_size, remain_delet=True, shuffle=True)
        self.edge_iter_valid = EdgeSampler(self.test_edge, batch_size, remain_delet=False, shuffle=True)
        self.neg_sampler = RandomNegativeSampler(self.para.neg_num)

    def _next_sample_public(self, edge_iter):
        src_ids, src_y = edge_iter.next()
        src_nbrs = self.neigbor_sampler.get(self.graph, src_ids)
        dest_nbrs = self.neigbor_sampler.get(self.graph, src_y)
        neg_batch = self.neg_sampler.get(self.graph, src_ids)
        neg_ids = neg_batch.ids.reshape(-1)
        neg_nbrs = self.neigbor_sampler.get(self.graph, neg_ids)
        return src_ids, src_y, neg_ids, src_nbrs, dest_nbrs, neg_nbrs

    def stop(self):
        pass

    def next(self, mode='train'):
        if mode == 'train':
            res = self._feed_next_sample(self.edge_iter_train)
        elif mode == 'valid':
            res = self._feed_next_sample(self.edge_iter_valid)
        # elif mode == 'test':
        #     res = self._feed_next_sample(self.node_iter_test)
        else:
            raise SystemError
        return res

    def valid_next(self):
        res = self._feed_next_sample(self.node_iter_valid)
        return res

    def test_next(self):
        res = self._feed_next_sample_emb(self.node_iter)
        return res

    def gather_feat(self, neighbor):
        xx = []
        for i, neighb_ in enumerate(neighbor):
            if i == 0:
                feat = [self.X[index_].A for index_ in neighb_]
                feat = sp.csr_matrix(np.concatenate(feat, axis=0))
                feat = sparse_mx_to_torch_sparse_tensor(feat)
                xx.append(feat)
            else:
                # shape = neighb_.shape
                neigh_ids = neighb_.ids.flatten()
                feat = [self.X[index_].A for index_ in neigh_ids]
                feat = np.concatenate(feat, axis=0)
                xx.append(sparse_mx_to_torch_sparse_tensor(sp.csr_matrix(feat)))
        return xx

    def _feed_next_sample(self, edge_iter):
        src_id, src_y, neg_ids, src_nbrs, dest_nbrs, neg_nbrs = self._next_sample_public(edge_iter)
        feat_src = self.gather_feat([src_id] + src_nbrs)
        feat_dest = self.gather_feat([src_y] + dest_nbrs)
        feat_neg = self.gather_feat([neg_ids] + neg_nbrs)
        # src_y = torch.LongTensor(src_y)
        return feat_src, feat_dest, feat_neg

    def _feed_next_sample_emb(self, node_iter):
        src_id = node_iter.next()
        src_nbrs = self.neigbor_sampler.get(self.graph, src_id)
        feat_src = self.gather_feat([src_id] + src_nbrs)
        return feat_src, src_id


class GraphInput_GCR(object):
    """
    An Input Manager that iteratively generate input feeds for training
    It takes a bipartite graph as input,
    specifically, ui_edges, ii_edges, u_attrs, and i_attrs
    """

    def __init__(
            self, flag
    ):
        self.graph = None
        self.graph_value = None
        self.X = None
        self.Y = None
        self.num_class = None
        self.train_index = None
        self.valid_index = None
        self.test_index = None
        self.node_num = None
        self.node_dim = None
        self.node_iter = None
        self.node_iter_valid = None
        self.node_iter_test = None
        self.neigbor_sampler = None
        self.neigbor_sampler_bi = None
        self.n_attr_graph = None
        self.attr_n_graph = None
        self.batch_size = flag.batch_size
        self.dataset = flag.dataset
        self.use_feat = flag.use_feat
        self.para = flag
        self.u_node_iter = None
        self.u_neighs_num = [] if flag.layer_num == '' else [int(x) for x in flag.layer_num.split(',')]
        self.u_neighs_num_bi = [] if flag.layer_num_bi == '' else [int(x) for x in flag.layer_num_bi.split(',')]
        self.u_depth = len(self.u_neighs_num)
        self.u_depth_bi = len(self.u_neighs_num_bi)
        self.avg_neigh = None
        self.avg_neigh_n = None
        self.avg_neigh_attr = None
        self.read_data_from_edges()
        self.u_neighs_num = [self.avg_neigh + 1 for i in range(self.u_depth)]
        self.u_neighs_num_bi = self.layer_wise_neigh()
        print('---> Layer-wise average number of neighbor is {}'.format(self.u_neighs_num))

    def layer_wise_neigh(self):
        layer_neigh = []
        for i in range(self.u_depth_bi):
            if (i + 1) % 2 == 1:
                neigh_n = min(self.avg_neigh_n, self.u_neighs_num_bi[i])
                layer_neigh.append(neigh_n)
            else:
                neigh_n = min(self.avg_neigh_attr, self.u_neighs_num_bi[i])
                layer_neigh.append(neigh_n)
        return layer_neigh

    def generate_train_index(self):
        all_index = set(list(range(self.node_num)))
        test_index = set(list(self.test_index))
        valid_index = set(list(self.valid_index))
        train_index = all_index - test_index - valid_index
        self.train_index = np.array(list(train_index))

    def read_data_from_edges(self):
        print('--> Loading {} dataset...'.format(self.dataset))
        data_file = 'data/' + self.para.dataset + '/{}.pkl'.format(self.para.dataset)
        with open(data_file, 'rb') as f:
            graph_, self.graph_value, self.X, self.Y, self.train_index,\
            self.valid_index, self.test_index = pickle.load(f)

        self.node_num, self.node_dim = self.X.shape
        if self.use_feat == 0:
            self.X = sp.sparse.csr_matrix(np.eye(self.node_num))
            self.node_dim = self.X.shape[1]
        self.num_class = np.max(self.Y) + 1
        src, dest = graph_[0], graph_[1]
        self.graph = HomeGraph(src, dest)
        self.avg_neigh = int(self.graph.avg_neigh)

        node_attri_graph = self.X.tocoo()

        self.n_attr_graph = BiGraph(node_attri_graph.row, node_attri_graph.col, node_attri_graph.data)
        self.attr_n_graph = BiGraph(node_attri_graph.col, node_attri_graph.row, node_attri_graph.data)
        self.avg_neigh_n = int(self.n_attr_graph.avg_neigh)
        self.avg_neigh_attr = int(self.attr_n_graph.avg_neigh)

        if self.para.semi == 0:
            self.generate_train_index()

        print('{} n_n graph: node {} dim {} edges {} avg_num {}'.format(self.dataset, self.node_num, self.node_dim, graph_.shape[1], self.graph.avg_neigh))
        print('{} n_attri graph: node {} attri {}'
              ' edges {} avg_num_n {}'
              ' avg_num_att {}'.format(self.dataset, self.n_attr_graph.node_len, self.attr_n_graph.node_len,
                                    self.n_attr_graph.edge_len, self.n_attr_graph.avg_neigh, self.attr_n_graph.avg_neigh))

        print('train_set: {} valid_set: {} test_set: {}'.format(self.train_index.shape[0],
                                                                self.valid_index.shape[0], self.test_index.shape[0]))

    def init_server(self):
        batch_size = self.batch_size
        self.neigbor_sampler = RandomNHopNeighborSampler_homoge(self.u_neighs_num)
        self.neigbor_sampler_bi = RandomNHopNeighborSampler_BI(self.u_neighs_num_bi)
        self.node_iter = NodeSampler(self.train_index, self.Y, batch_size, remain_delet=True, shuffle=True)
        self.node_iter_valid = NodeSampler(self.valid_index, self.Y, batch_size, remain_delet=False, shuffle=False)
        self.node_iter_test = NodeSampler(self.test_index, self.Y, batch_size, remain_delet=False, shuffle=False)

    def _next_sample_public(self, node_iter):
        src_ids, src_y = node_iter.next()
        src_nbrs = self.neigbor_sampler.get(self.graph, src_ids)
        src_nbrs_bi = self.neigbor_sampler_bi.get(interleaving_seq(self.n_attr_graph, self.attr_n_graph, self.u_depth_bi),
                                          src_ids, with_attr=False)
        return src_ids, src_y, src_nbrs, src_nbrs_bi

    def stop(self):
        pass

    def next(self, mode='train'):
        if mode == 'train':
            res = self._feed_next_sample(self.node_iter)
        elif mode == 'valid':
            res = self._feed_next_sample(self.node_iter_valid)
        elif mode == 'test':
            res = self._feed_next_sample(self.node_iter_test)
        else:
            raise SystemError
        return res

    def valid_next(self):
        res = self._feed_next_sample(self.node_iter_valid)
        return res

    def test_next(self):
        res = self._feed_next_sample(self.node_iter_test)
        return res

    def gather_feat(self, neighbor):
        xx = []
        for i, neighb_ in enumerate(neighbor):
            if i == 0:
                feat = [self.X[index_].A for index_ in neighb_]
                feat = sp.sparse.csr_matrix(np.concatenate(feat, axis=0))
                feat = sparse_mx_to_torch_sparse_tensor(feat)
                xx.append(feat)
            else:
                # shape = neighb_.shape
                neigh_ids = neighb_.ids.flatten()
                feat = [self.X[index_].A for index_ in neigh_ids]
                feat = np.concatenate(feat, axis=0)
                xx.append(sparse_mx_to_torch_sparse_tensor(sp.sparse.csr_matrix(feat)))
        return xx

    def gather_feat_bi(self, neighbor):
        xx = []
        for i in range(len(neighbor)):
            if (i + 1) % 2 == 1:
                yy = torch.LongTensor(neighbor[i].ids)
                xx.append(yy)
            else:
                neigh_ids = neighbor[i].ids.flatten()
                feat = [self.X[index_].A for index_ in neigh_ids]
                feat = np.concatenate(feat, axis=0)
                xx.append(sparse_mx_to_torch_sparse_tensor(sp.sparse.csr_matrix(feat)))

        return xx

    def _feed_next_sample(self, node_iter):
        src_id, src_y, hop_neigh, src_nbrs_bi = self._next_sample_public(node_iter)
        feat_list = self.gather_feat([src_id] + hop_neigh)
        feat_list_bi = self.gather_feat_bi(src_nbrs_bi)
        src_y = torch.LongTensor(src_y)
        return feat_list, src_y, [feat_list[0]] + feat_list_bi


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--cuda', type=str, default='6', help='specify cuda devices')
    parser.add_argument('--dataset', type=str, default="cora",
                        help='Dataset to use.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--layer_num', type=str, default='4,10', help='adam or sgd')
    parser.add_argument('--layer_num_bi', type=str, default='15,10', help='adam or sgd')
    parser.add_argument('--model', type=str, default='gcr', help='adam or sgd')
    parser.add_argument('--hidden', type=str, default='16,16', help='adam or sgd')
    parser.add_argument('--hidden_bi', type=str, default='16,16', help='adam or sgd')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--use_feat', type=int, default=1,
                        help='Use feat or not')
    parser.add_argument('--semi', type=int, default=1,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--normalization', type=str, default='AugNormAdj',
                        choices=['AugNormAdj'],
                        help='Normalization method for the adjacency matrix.')
    parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')

    args = parser.parse_args()
    # data_loader = GraphInput(args)
    data_loader = GraphInput_GCR(args)
    data_loader.init_server()
    batch_sample = data_loader.next()
    print('finished')



