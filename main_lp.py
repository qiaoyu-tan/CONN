from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import scipy.io as sio
import pickle
import os
import torch
import torch.optim as optim
import scipy.sparse as sp
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from optimizer import loss_function_entropysample

from utils import load_citationANEWeight, sparse_mx_to_torch_sparse_tensor, EdgeSampler, load_citationANEmatWeight, normalize, normalize_array
from gcl.model import CONN
from preprocessing import mask_test_edges_bipartall, mask_test_edges_net
from gcl.fintune import finetune_GCLWeightGNN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=str, default='2', help='specify cuda devices')
parser.add_argument('--dataset', type=str, default="cora",
                    help='Dataset to use.')
parser.add_argument('--model_type', type=str, default="conn_lp",
                    help='Dataset to use.')
parser.add_argument('--mode', type=str, default="train",
                    help='Dataset to use.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--nlayer', type=int, default=2, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--activate', type=str, default="relu",
                    help='relu | prelu')
parser.add_argument('--batch_size', type=int, default=1024,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--trade_weight', type=float, default=0.8,
                    help='trade_off parameters).')
parser.add_argument('--hid1', type=int, default=512,
                    help='Number of hidden units.')
parser.add_argument('--hid2', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dim', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--use_cpu', type=int, default=0,
                    help='Use attribute or not')
parser.add_argument('--loss_type', type=str, default="entropy",
                    help='entropy | BPR')
parser.add_argument('--patience', type=int, default=50,
                    help='Use attribute or not')
parser.add_argument('--dropout', type=float, default=0.6,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--drop', type=int, default=1,
                    help='Indicate whether drop out or not')
parser.add_argument('--save_num', type=int, default=20,
                    help='Use attribute or not')

def get_roc_score(net, adj, dataloader_val, device):
    net.eval()
    preds = []
    preds_neg = []
    while True:
        try:
            pos_edge, neg_edge = dataloader_val.next()
        except StopIteration:
            break
        pos_src_, pos_dst_ = zip(*pos_edge)
        neg_src_, neg_dst_ = zip(*neg_edge)
        pos_src = torch.LongTensor(pos_src_).to(device)
        pos_dst = torch.LongTensor(pos_dst_).to(device)
        neg_src = torch.LongTensor(neg_src_).to(device)
        neg_dst = torch.LongTensor(neg_dst_).to(device)
        src_emb, dst_emb, src_neg_emb, dst_neg_emb = net(adj, pos_src, pos_dst,
                                                           neg_src, neg_dst)
        pos_logit, neg_logit = net.pred_logits(src_emb, dst_emb, src_neg_emb, dst_neg_emb)
        pos_logit = torch.sigmoid(pos_logit)
        neg_logit = torch.sigmoid(neg_logit)
        pos_logit = pos_logit.data.cpu().numpy().reshape(-1)
        neg_logit = neg_logit.data.cpu().numpy().reshape(-1)
        preds.extend(pos_logit.tolist())
        preds_neg.extend(neg_logit.tolist())

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)).tolist(), np.zeros(len(preds_neg)).tolist()])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

def output_nodeemb(adj, net, num_node, device):
    net.eval()
    batch_size = 64
    n = int(num_node / batch_size)
    if n * batch_size < num_node:
        n = n + 1
    index_all = list(range(num_node))
    start = 0
    x1 = []
    for i in range(n):
        if i == n-1:
            index_ = np.array(index_all[start:])
        else:
            index_ = np.array(index_all[start:start+batch_size])

        node_index = torch.LongTensor(index_).to(device)
        node_emb = net.get_emb(node_index, adj)
        x1.append(node_emb.data.cpu().numpy())
        start += batch_size
    x1 = np.concatenate(x1, axis=0)
    return x1


def train(features, adj, dataloader, dataloader_val, dataloader_test, save_path, device, args, pos_weight, norm):
    num_node = features.shape[0]
    num_attri = features.shape[1]
    model = CONN(nfeat=args.dim,
                         nlayer=args.nlayer,
                         nnode=num_node,
                         nattri=num_attri,
                         dropout=args.dropout,
                         drop=args.drop,
                         hid1=args.hid1,
                         hid2=args.hid2,
                         act=args.activate)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    pos_weight = pos_weight.to(device)
    b_xent = nn.BCEWithLogitsLoss()

    model.to(device)
    adj = adj.to(device)
    max_auc = 0.0
    max_ap = 0.0
    best_epoch = 0
    cnt_wait = 0
    for epoch in range(args.epochs):
        steps = 0
        epoch_loss = 0.0
        model.train()
        while True:
            try:
                pos_edge, neg_edge = dataloader.next()
            except StopIteration:
                break
            pos_src_, pos_dst_ = zip(*pos_edge)
            neg_src_, neg_dst_ = zip(*neg_edge)
            pos_src = torch.LongTensor(pos_src_).to(device)
            pos_dst = torch.LongTensor(pos_dst_).to(device)
            neg_src = torch.LongTensor(neg_src_).to(device)
            neg_dst = torch.LongTensor(neg_dst_).to(device)
            src_emb, dst_emb, src_neg_emb, dst_neg_emb = model(adj, pos_src, pos_dst, neg_src, neg_dst)
            pos_logit, neg_logit = model.pred_logits(src_emb, dst_emb, src_neg_emb, dst_neg_emb)
            loss_train = loss_function_entropysample(pos_logit, neg_logit, b_xent, loss_type=args.loss_type)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            epoch_loss += loss_train.item()
            print('--> Epoch %d Step %5d loss: %.3f' % (epoch + 1, steps + 1, loss_train.item()))
            steps += 1

        auc_, ap_ = get_roc_score(model, adj, dataloader_val, device)
        if auc_ > max_auc:
            max_auc = auc_
            max_ap = ap_
            best_epoch = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), save_path)
        else:
            cnt_wait += 1

        print('Epoch %d / %d' % (epoch, args.epochs),
              'current_best_epoch: %d' % best_epoch,
              'train_loss: %.4f' % (epoch_loss / steps),
              'valid_acu: %.4f' % auc_,
              'valid_ap: %.4f' % ap_)

        if cnt_wait == args.patience:
            print('Early stopping!')
            break

    print('!!! Training finished',
          'best_epoch: %d' % best_epoch,
          'best_auc: %.4f' % max_auc,
          'best_ap: %.4f' % max_ap)

    model.load_state_dict(torch.load(save_path))

    emb = output_nodeemb(adj, model, num_node, device)
    return emb

def print_configuration(args):
    print('--> Experiment configuration')
    for key, value in vars(args).items():
        print('{}: {}'.format(key, value))


def hop1_get(adj, trade_weight, num_node):
    adj = adj.tolil()
    adj_attr = adj[:num_node, num_node:].tocsr()
    adj_net = adj[:num_node, :num_node].tocsr()
    adj_attr = normalize(adj_attr)
    adj_net = normalize(adj_net)
    n, d = adj_attr.shape
    adj_train = sp.dok_matrix((n + d, n + d), dtype=np.float32)
    adj_train = adj_train.tolil()
    weight_net = trade_weight
    weight_attr = (1 - trade_weight)
    adj_net = adj_net * weight_net
    adj_net = adj_net.tolil()
    adj_attr = adj_attr * weight_attr
    adj_attr = adj_attr.tolil()
    adj_attri_2 = sp.csr_matrix(np.eye(d, dtype=float)).tolil()
    adj_attri_2 = adj_attri_2 * weight_net
    adj_train[:n, n:] = adj_attr
    adj_train[n:, :n] = adj_attr.T
    adj_train[:n, :n] = adj_net
    adj_train[n:, n:] = adj_attri_2
    adj_train = adj_train.tocsr()
    return adj_train

if __name__ == '__main__':
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        cuda_name = 'cuda:' + args.cuda
        device = torch.device(cuda_name)
        print('--> Use GPU %s' % args.cuda)
        torch.cuda.manual_seed(args.seed)
    else:
        device = torch.device("cpu")
        print("--> No GPU")

    if args.use_cpu:
        device = torch.device("cpu")

    print('---> Loading %s dataset...' % args.dataset)
    if args.dataset == 'BlogCatalog' or args.dataset == 'Flickr' or args.dataset == 'ACM':
        adj_ori, features, adj, labels, idx_train, idx_val, idx_test = load_citationANEmatWeight(args.dataset)
    else:
        adj_ori, features, adj, labels, idx_train, idx_val, idx_test = load_citationANEWeight(args.dataset)
    print('--->Generate train/valid links for unsupervised learning...')
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    num_node = features.shape[0]

    print('---> Prepare training loader...')
    t1 = time.time()
    adj_train_, train_edges_, train_edges_false_, val_edges_, val_edges_false_, test_edges, test_edges_false = mask_test_edges_net(adj_ori)
    t2 = time.time()
    adj_train, train_edges, val_edges, val_edges_false, test_edges_, test_edges_false_ = mask_test_edges_bipartall(adj,features.shape[0])
    adj_train = adj_train.tolil()
    adj_train[:num_node, :num_node] = adj_train_.tolil()
    adj_train = adj_train.tocsr()
    train_edges[0] = np.concatenate([train_edges[0], train_edges_], axis=0)
    train_edges[1] = np.concatenate([train_edges[1], train_edges_false_], axis=0)
    val_edges = np.concatenate([val_edges, val_edges_], axis=0)
    val_edges_false = np.concatenate([val_edges_false, val_edges_false_], axis=0)

    adj = adj_train
    print('---> Finish training loader with time: %d' % (time.time() - t1))

    idx_train = np.array(idx_train)
    idx_val = np.array(idx_val)
    idx_test = np.array(idx_test)

    mat_path = 'emb/lp-' + args.dataset + '_{}'.format(args.trade_weight) + 'gnn.mat'
    if os.path.isfile(mat_path):
        with open(mat_path, 'rb') as f:
            data = pickle.load(f)
            adj_norm = data['adj_norm']
    else:
        print('---> Start adj_norm')
        tt1 = time.time()
        adj_norm = hop1_get(adj, args.trade_weight, num_node)
        tt2 = time.time()
        print('---> finish adj_norm with time: {}'.format(tt2 - tt1))
        save_dict = {'adj_norm': adj_norm}

        with open(mat_path, 'wb') as pfile:
            pickle.dump(save_dict, pfile, pickle.HIGHEST_PROTOCOL)

    dataloader = EdgeSampler(train_edges[0], train_edges[1], args.batch_size)
    dataloader_val = EdgeSampler(val_edges, np.array(val_edges_false), args.batch_size, remain_delet=False)
    dataloader_test = EdgeSampler(test_edges, np.array(test_edges_false), args.batch_size, remain_delet=False)
    pos_weight = torch.Tensor([float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()])
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm).float()

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    save_path = "./weights/linkwet_%s_" % args.model_type + args.dataset + '%d' % args.nlayer + '_%d_' % args.dim + '_%d_' % args.hid1 + '_%d_' % args.hid2 + '{}'.format(args.trade_weight)+ '.pth'
    print_configuration(args)

    print('---> Start training...')
    node_emb = train(features, adj_norm, dataloader, dataloader_val, dataloader_test, save_path, device, args,
                          pos_weight, norm)
    features = sp.csr_matrix(node_emb)
    t1 = time.time()
    auc_gcl, ap_gcl = finetune_GCLWeightGNN(features, device, args)
    t2 = time.time()
    print('!!! Link prediction results for ',
          'auc_gcl: %.4f' % (auc_gcl),
          'ap_gcl: %.4f' % ap_gcl
          )
    print('!!! Finish')



