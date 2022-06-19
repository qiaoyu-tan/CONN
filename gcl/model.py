import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, InnerProductDecoder
import torch

class CONN(nn.Module):
    def __init__(self, nfeat, nnode, nattri, nlayer, dropout, drop, hid1=512, hid2=128, act='relu'):
        super(CONN, self).__init__()
        self.latent_dim = nfeat
        self.decoder = InnerProductDecoder(nfeat, dropout)
        self.dropout = dropout
        self.nlayer = nlayer
        self.drop = drop
        self.hid1 = hid1
        self.hid2 = hid2
        if act == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = nn.PReLU()

        layer = []
        for i in range(self.nlayer):
            layer.append(GraphConvolution(nfeat, nfeat))
        self.gc1 = nn.ModuleList(layer)

        self.num_node = nnode
        self.num_attri = nattri
        self.embedding_node = torch.nn.Embedding(
            num_embeddings=self.num_node, embedding_dim=self.latent_dim)
        self.embedding_attri = torch.nn.Embedding(
            num_embeddings=self.num_attri, embedding_dim=self.latent_dim)

        # prediction layer
        n_layer = (self.nlayer + 1) * (self.nlayer + 1)
        self.mlp1 = nn.Linear(n_layer * self.latent_dim, self.hid1, bias=False)
        self.mlp2 = nn.Linear(self.hid1, self.hid2, bias=False)
        self.mlp3 = nn.Linear(self.hid2, 1, bias=True)
        self.reset_parameters()

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def reset_parameters(self):
        torch.nn.init.normal_(self.embedding_node.weight, std=0.1)
        torch.nn.init.normal_(self.embedding_attri.weight, std=0.1)

    def forward(self, adj, pos_src, pos_dst, neg_src, neg_dst):
        if self.training:
            if self.drop:
                adj = self.__dropout(adj)
        x1 = torch.cat([self.embedding_node.weight, self.embedding_attri.weight], dim=0)
        src_emb = []
        dst_emb = []
        src_neg_emb = []
        dst_neg_emb = []
        src_emb.append(F.normalize(x1[pos_src], p=2, dim=1))
        dst_emb.append(F.normalize(x1[pos_dst], p=2, dim=1))
        src_neg_emb.append(F.normalize(x1[neg_src], p=2, dim=1))
        dst_neg_emb.append(F.normalize(x1[neg_dst], p=2, dim=1))

        for i, layer in enumerate(self.gc1):
            x1 = layer(x1, adj)
            src_emb.append(F.normalize(x1[pos_src], p=2, dim=1))
            dst_emb.append(F.normalize(x1[pos_dst], p=2, dim=1))
            src_neg_emb.append(F.normalize(x1[neg_src], p=2, dim=1))
            dst_neg_emb.append(F.normalize(x1[neg_dst], p=2, dim=1))

        return src_emb, dst_emb, src_neg_emb, dst_neg_emb

    def comute_hop_emb(self, src_adj, dst_adj, src_neg_adj, dst_neg_adj):
        if self.training:
            if self.drop:
                src_adj = self.__dropout(src_adj)
                dst_adj = self.__dropout(dst_adj)
                src_neg_adj = self.__dropout(src_neg_adj)
                dst_neg_adj = self.__dropout(dst_neg_adj)
        x1 = torch.cat([self.embedding_node.weight, self.embedding_attri.weight], dim=0)

        src_emb_2 = self.gc2(x1, src_adj)
        dst_emb_2 = self.gc2(x1, dst_adj)
        src_neg_emb_2 = self.gc2(x1, src_neg_adj)
        dst_neg_emb_2 = self.gc2(x1, dst_neg_adj)
        return [src_emb_2], [dst_emb_2], [src_neg_emb_2], [dst_neg_emb_2]

    def get_emb(self, node_index, adj):
        x1 = torch.cat([self.embedding_node.weight, self.embedding_attri.weight], dim=0)
        node_emb = []
        node_emb.append(F.normalize(x1[node_index], p=2, dim=1))

        for i, layer in enumerate(self.gc1):
            x1 = layer(x1, adj)
            node_emb.append(F.normalize(x1[node_index], p=2, dim=1))

        return node_emb[1]

    def bi_cross_layer(self, x_1, x_2):
        bi_layer = []
        for i in range(len(x_1)):
            xi = x_1[i]
            for j in range(len(x_2)):
                xj = x_2[j]
                bi_layer.append(torch.mul(xi, xj))
        return bi_layer

    def cross_layer(self, src_x, dst_x):
        bi_layer = self.bi_cross_layer(src_x, dst_x)
        bi_layer = torch.cat(bi_layer, dim=1)
        return bi_layer

    def compute_logits(self, emb):
        emb = self.mlp1(emb)
        emb = self.act(emb)
        emb = self.mlp2(emb)
        emb = self.act(emb)
        preds = self.mlp3(emb)
        return preds

    def pred_logits(self, src_emb, dst_emb, src_neg_emb, dst_neg_emb):
        emb_pos = self.cross_layer(src_emb, dst_emb)
        emb_neg = self.cross_layer(src_neg_emb, dst_neg_emb)
        logits_pos = self.compute_logits(emb_pos)
        logits_neg = self.compute_logits(emb_neg)
        return logits_pos, logits_neg

    def pred_score(self, input_emb):
        preds = self.decoder(input_emb)
        return torch.sigmoid(preds)

    def __dropout(self, graph):
        graph = self.__dropout_x(graph)
        return graph

    def __dropout_x(self, x):
        x = x.coalesce()
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + self.dropout
        # random_index = random_index.int().bool()
        random_index = random_index.int().type(torch.bool)

        index = index[random_index]
        # values = values[random_index]/self.dropout
        values = values[random_index]
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def reg_loss(self):
        reg_loss = (1 / 2) * (self.embedding_node.weight.norm(2).pow(2) +
                              self.embedding_attri.weight.norm(2).pow(2) / float(self.num_node + self.num_attri))
        return reg_loss


class connTune(nn.Module):
    def __init__(self, nfeat, nnode, nattri, nlayer, dropout, drop, hid1=512, hid2=128, act='relu'):
        super(connTune, self).__init__()
        self.latent_dim = nfeat
        self.decoder = InnerProductDecoder(nfeat, dropout)
        self.dropout = dropout
        self.nlayer = nlayer
        self.drop = drop
        self.hid1 = hid1
        self.hid2 = hid2
        if act == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = nn.PReLU()

        layer = []
        for i in range(self.nlayer):
            layer.append(GraphConvolution(nfeat, nfeat))
        self.gc1 = nn.ModuleList(layer)
        self.num_node = nnode
        self.num_attri = nattri
        n_layer = (self.nlayer + 1) * (self.nlayer + 1)
        self.mlp1 = nn.Linear(n_layer * self.latent_dim, self.hid1, bias=False)
        self.mlp2 = nn.Linear(self.hid1, self.hid2, bias=False)
        self.mlp3 = nn.Linear(self.hid2, 1, bias=True)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def reset_parameters(self):
        torch.nn.init.normal_(self.embedding_node.weight, std=0.1)
        torch.nn.init.normal_(self.embedding_attri.weight, std=0.1)

    def forward(self, x1, adj, pos_src, pos_dst, neg_src, neg_dst):
        if self.training:
            if self.drop:
                adj = self.__dropout(adj)
        src_emb = []
        dst_emb = []
        src_neg_emb = []
        dst_neg_emb = []
        src_emb.append(F.normalize(x1[pos_src], p=2, dim=1))
        dst_emb.append(F.normalize(x1[pos_dst], p=2, dim=1))
        src_neg_emb.append(F.normalize(x1[neg_src], p=2, dim=1))
        dst_neg_emb.append(F.normalize(x1[neg_dst], p=2, dim=1))

        for i, layer in enumerate(self.gc1):
            x1 = layer(x1, adj)
            src_emb.append(F.normalize(x1[pos_src], p=2, dim=1))
            dst_emb.append(F.normalize(x1[pos_dst], p=2, dim=1))
            src_neg_emb.append(F.normalize(x1[neg_src], p=2, dim=1))
            dst_neg_emb.append(F.normalize(x1[neg_dst], p=2, dim=1))

        return src_emb, dst_emb, src_neg_emb, dst_neg_emb

    def comute_hop_emb(self, src_adj, dst_adj, src_neg_adj, dst_neg_adj):
        if self.training:
            if self.drop:
                src_adj = self.__dropout(src_adj)
                dst_adj = self.__dropout(dst_adj)
                src_neg_adj = self.__dropout(src_neg_adj)
                dst_neg_adj = self.__dropout(dst_neg_adj)
        x1 = torch.cat([self.embedding_node.weight, self.embedding_attri.weight], dim=0)

        src_emb_2 = self.gc2(x1, src_adj)
        dst_emb_2 = self.gc2(x1, dst_adj)
        src_neg_emb_2 = self.gc2(x1, src_neg_adj)
        dst_neg_emb_2 = self.gc2(x1, dst_neg_adj)
        return [src_emb_2], [dst_emb_2], [src_neg_emb_2], [dst_neg_emb_2]

    def get_emb(self, x1, node_index, adj):
        node_emb = []
        node_emb.append(F.normalize(x1[node_index], p=2, dim=1))
        for i, layer in enumerate(self.gc1):
            x1 = layer(x1, adj)
            node_emb.append(F.normalize(x1[node_index], p=2, dim=1))

        return node_emb

    def get_emb2(self, adj):
        xx = torch.cat([self.embedding_node.weight, self.embedding_attri.weight], dim=0)
        node_emb = self.gc2(xx, adj)
        return node_emb

    def bi_cross_layer(self, x_1, x_2):
        bi_layer = []
        for i in range(len(x_1)):
            xi = x_1[i]
            for j in range(len(x_2)):
                xj = x_2[j]
                bi_layer.append(torch.mul(xi, xj))
        return bi_layer

    def cross_layer(self, src_x, dst_x):
        bi_layer = self.bi_cross_layer(src_x, dst_x)
        bi_layer = torch.cat(bi_layer, dim=1)
        return bi_layer

    def compute_logits(self, emb):
        emb = self.mlp1(emb)
        emb = self.act(emb)
        emb = self.mlp2(emb)
        emb = self.act(emb)
        preds = self.mlp3(emb)
        return preds

    def pred_logits(self, src_emb, dst_emb, src_neg_emb, dst_neg_emb):
        emb_pos = self.cross_layer(src_emb, dst_emb)
        emb_neg = self.cross_layer(src_neg_emb, dst_neg_emb)
        logits_pos = self.compute_logits(emb_pos)
        logits_neg = self.compute_logits(emb_neg)
        return logits_pos, logits_neg

    def pred_score(self, input_emb):
        preds = self.decoder(input_emb)
        return torch.sigmoid(preds)

    def __dropout(self, graph):
        graph = self.__dropout_x(graph)
        return graph

    def __dropout_x(self, x):
        x = x.coalesce()
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + self.dropout
        # random_index = random_index.int().bool()
        random_index = random_index.int().type(torch.bool)

        index = index[random_index]
        # values = values[random_index]/self.dropout
        values = values[random_index]
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def reg_loss(self):
        reg_loss = (1 / 2) * (self.embedding_node.weight.norm(2).pow(2) +
                              self.embedding_attri.weight.norm(2).pow(2) / float(self.num_node + self.num_attri))
        return reg_loss