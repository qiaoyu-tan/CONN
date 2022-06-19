import math
import torch.nn.functional as F
import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            torch.nn.init.xavier_uniform_(self.bias)
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphConvolutionCONN(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, agg=1, bias=False):
        super(GraphConvolutionCONN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_weight = agg
        if self.use_weight:
            self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            torch.nn.init.xavier_uniform_(self.bias)
        if self.use_weight:
            torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        if self.use_weight:
            support = torch.mm(input, self.weight)
            output = torch.spmm(adj, support)
        else:
            output = torch.sparse.mm(adj, input)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCNTrans(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GCNTrans, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class NeighLayer(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, agg_type, bias=False):
        super(NeighLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.agg_type = agg_type
        if self.agg_type == 'atten':
            self.weight = Parameter(torch.FloatTensor(in_features, 1))
        elif self.agg_type == 'mean':
            pass
        elif self.agg_type == 'sum':
            pass

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            torch.nn.init.xavier_uniform_(self.bias)
        if self.agg_type == 'atten':
            torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        if self.agg_type == 'mean':
            output = torch.sparse.mm(adj, input)
            weight = torch.sparse.sum(adj, dim=1).to_dense().view(-1, 1)
            output = output / weight
            output = torch.where(torch.isnan(output), torch.full_like(output, 0), output)
            # support = torch.mm(input, self.weight)
        elif self.agg_type == 'sum':
            output = torch.sparse.mm(adj, input)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class InnerProductDecoder(Module):
    """Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., act=F.sigmoid, **kwargs):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act
        self.dim = input_dim

    def forward(self, inputs):
        inputs = F.dropout(inputs, self.dropout, training=self.training)
        outputs = torch.mm(inputs, inputs.t())
        # x = tf.reshape(x, [-1])
        return outputs

class SparseGraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(SparseGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            torch.nn.init.xavier_uniform_(self.bias)
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        support = torch.spmm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class SparseGCN(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(SparseGCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            torch.nn.init.xavier_uniform_(self.bias)
        # torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        support = torch.spmm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class SparseLayer(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(SparseLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, self_vecs, neigh_vecs, neigh_num):
        self_vecs = torch.spmm(self_vecs, self.weight)
        self_vecs = self_vecs.view(-1, 1, self.out_features)

        neigh_vecs = torch.spmm(neigh_vecs, self.weight)
        neigh_vecs = neigh_vecs.view(-1, neigh_num, self.out_features)
        output = torch.cat([self_vecs, neigh_vecs], dim=1)  # [batch_size, neigh_num+1, dim]

        output = torch.mean(output, dim=1)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class DenseLayer(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(DenseLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, self_vecs, neigh_vecs, neigh_num):
        self_vecs = torch.mm(self_vecs, self.weight)
        self_vecs = self_vecs.view(-1, 1, self.out_features)

        neigh_vecs = torch.mm(neigh_vecs, self.weight)
        neigh_vecs = neigh_vecs.view(-1, neigh_num, self.out_features)
        output = torch.cat([self_vecs, neigh_vecs], dim=1)

        output = torch.mean(output, dim=1)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

