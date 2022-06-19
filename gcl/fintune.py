import torch
import torch.optim as optim
import scipy.sparse as sp
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import numpy as np
from gcl.model import connTune
from optimizer import loss_function_entropysample
from utils import normalize, sparse_mx_to_torch_sparse_tensor, EdgeSampler, load_citationANEmat, load_citation
from preprocessing import mask_test_edges_net_lp
from process import sparse_mx_to_torch_sparse_tensor

def accuracy(preds, labels):
    # correct = preds.eq(labels).double()
    correct = (preds == labels).astype(float)
    correct = correct.sum()
    return correct / len(labels)

def lookup_adj(pos_src, pos_dst, neg_src, neg_dst, adj, device):
    src_adj = adj[np.array(pos_src)]
    dst_adj = adj[np.array(pos_dst)]
    src_neg_adj = adj[np.array(neg_src)]
    dst_neg_adj = adj[np.array(neg_dst)]
    if type(adj) is np.ndarray:
        src_adj = sp.csr_matrix(src_adj)
        dst_adj = sp.csr_matrix(dst_adj)
        src_neg_adj = sp.csr_matrix(src_neg_adj)
        dst_neg_adj = sp.csr_matrix(dst_neg_adj)
    src_adj = sparse_mx_to_torch_sparse_tensor(src_adj).to(device)
    dst_adj = sparse_mx_to_torch_sparse_tensor(dst_adj).to(device)
    src_neg_adj = sparse_mx_to_torch_sparse_tensor(src_neg_adj).to(device)
    dst_neg_adj = sparse_mx_to_torch_sparse_tensor(dst_neg_adj).to(device)
    return src_adj, dst_adj, src_neg_adj, dst_neg_adj

def get_roc_score_cgnn(net, adj, features, dataloader_val, device):
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
        src_emb, dst_emb, src_neg_emb, dst_neg_emb = net(features, adj, pos_src, pos_dst, neg_src, neg_dst)

        pos_logit, neg_logit = net.pred_logits(src_emb, dst_emb,
                                                 src_neg_emb, dst_neg_emb)
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


def get_roc_score(net, features, adj, adj2, dataloader_val, device):
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
        src_adj, dst_adj, src_neg_adj, dst_neg_adj = lookup_adj(pos_src_, pos_dst_, neg_src_, neg_dst_, adj, device)
        pos_src = torch.LongTensor(pos_src_).to(device)
        pos_dst = torch.LongTensor(pos_dst_).to(device)
        neg_src = torch.LongTensor(neg_src_).to(device)
        neg_dst = torch.LongTensor(neg_dst_).to(device)
        src_emb, dst_emb, src_neg_emb, dst_neg_emb = net(features, src_adj, dst_adj, src_neg_adj, dst_neg_adj, pos_src, pos_dst,
                                                           neg_src, neg_dst)

        src_adj, dst_adj, src_neg_adj, dst_neg_adj = lookup_adj(pos_src_, pos_dst_, neg_src_, neg_dst_, adj2, device)
        src_emb_, dst_emb_, src_neg_emb_, dst_neg_emb_ = net.comute_hop_emb(features, src_adj, dst_adj, src_neg_adj,
                                                                              dst_neg_adj)
        pos_logit, neg_logit = net.pred_logits(src_emb + src_emb_, dst_emb + dst_emb_,
                                                 src_neg_emb + src_neg_emb_, dst_neg_emb + dst_neg_emb_)
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


def get_roc_score_conn(net, features, adj, adj2, dataloader_val, device):
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
        src_emb, dst_emb, src_neg_emb, dst_neg_emb = net(features, [adj, adj2], pos_src, pos_dst,
                                                           neg_src, neg_dst)
        pos_logit, neg_logit = net.pred_logits(src_emb, dst_emb,
                                                 src_neg_emb, dst_neg_emb)
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

def get_roc_scoreGNN(net, features, adj, dataloader_val, device):
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
        src_emb, dst_emb, src_neg_emb, dst_neg_emb = net(features, adj, pos_src, pos_dst, neg_src, neg_dst)

        pos_logit, neg_logit = net.pred_logits(src_emb, dst_emb,
                                                 src_neg_emb, dst_neg_emb)
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

def output_nodeembGNN(features, adj, net, num_node, device, n=20):
    net.eval()
    # batch_size = int(num_node / n)
    batch_size = 64
    n = int(num_node / batch_size)
    if n * batch_size < num_node:
        n = n + 1
    index_all = list(range(num_node))
    start = 0
    x1 = []
    x2 = []
    x3 = []
    for i in range(n):
        if i == n-1:
            index_ = np.array(index_all[start:])
        else:
            index_ = np.array(index_all[start:start+batch_size])

        node_index = torch.LongTensor(index_).to(device)
        node_emb = net.get_emb(features, node_index, adj)
        x1.append(node_emb[0].data.cpu().numpy())
        x2.append(node_emb[1].data.cpu().numpy())
        x3.append(node_emb[2].data.cpu().numpy())
        start += batch_size
    x1 = np.concatenate(x1, axis=0)
    x2 = np.concatenate(x2, axis=0)
    x3 = np.concatenate(x3, axis=0)
    return [x1, x2, x3]

def train_GCLWeightGNN(features, adj, dataloader, dataloader_val, save_path, device, args, pos_weight, norm):
    num_node = features.shape[0]
    num_attri = features.shape[1]
    model = GCRANEWeightGNNTune(nfeat=args.dim,
                            emb_norm=args.emb_norm_tune,
                            nnode=num_node,
                            nattri=num_attri,
                            nlayer=args.nlayer,
                            nonlinear=args.nonlinear,
                            dropout=args.dropout,
                            drop=args.drop,
                            hid1=args.hid1,
                            hid2=args.hid2,
                            act=args.activate)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    b_xent = nn.BCEWithLogitsLoss()

    model.to(device)
    adj = adj.to(device)
    features = features.to(device)
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
            src_emb, dst_emb, src_neg_emb, dst_neg_emb = model(features, adj, pos_src, pos_dst, neg_src, neg_dst)

            pos_logit, neg_logit = model.pred_logits(src_emb, dst_emb,
                                                     src_neg_emb, dst_neg_emb)
            loss_train = loss_function_entropysample(pos_logit, neg_logit, b_xent, loss_type=args.loss_type)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            epoch_loss += loss_train.item()
            print('--> Epoch %d Step %5d loss: %.3f' % (epoch + 1, steps + 1, loss_train.item()))
            steps += 1

        auc_, ap_ = get_roc_scoreGNN(model, features, adj, dataloader_val, device)
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

    emb_result = []
    model.load_state_dict(torch.load(save_path))
    emb = output_nodeembGNN(features, adj, model, num_node, device, args.save_num)

    emb_result.append(emb[0])
    emb_result.append(emb[1])
    emb_result.append(emb[2])
    emb_result.append(emb[0] + emb[1])
    emb_result.append(emb[0] + emb[1] + emb[2])
    emb_result.append(emb[1] + emb[2])
    # emb_table = np.concatenate(emb[0:2], axis=1)
    # emb_result.append(emb_table)
    #
    # emb_table = np.concatenate(emb[0:3], axis=1)
    # emb_result.append(emb_table)
    #
    # emb_table = np.concatenate(emb[1:3], axis=1)
    # emb_result.append(emb_table)
    return emb_result

def output_nodeemb_cgnn(adj, net, features, num_node, device, n=20):
    net.eval()
    batch_size = int(num_node / n)
    index_all = list(range(num_node))
    start = 0
    x1 = []
    x2 = []
    x3 = []
    for i in range(n):
        if i == n-1:
            index_ = np.array(index_all[start:])
        else:
            index_ = np.array(index_all[start:start+batch_size])

        node_index = torch.LongTensor(index_).to(device)
        xx = net.get_emb(features, node_index, adj)
        x1.append(xx[0])
        x2.append(xx[1])
        x3.append(xx[2])
        start += batch_size
    x1 = torch.cat(x1, dim=0).data.cpu().numpy()
    x2 = torch.cat(x2, dim=0).data.cpu().numpy()
    x3 = torch.cat(x3, dim=0).data.cpu().numpy()
    return [x1, x2, x3]

def train_GCLGNN(features, adj, dataloader, dataloader_val, dataloader_test, save_path, device, args, pos_weight, norm):
    num_node = features.shape[0]
    num_attri = features.shape[1]
    model = connTune(nfeat=num_attri,
                                nnode=num_node,
                                nattri=num_attri,
                                nlayer=args.nlayer,
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
    features = features.to(device)
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
            src_emb, dst_emb, src_neg_emb, dst_neg_emb = model(features, adj, pos_src, pos_dst, neg_src, neg_dst)
            pos_logit, neg_logit = model.pred_logits(src_emb, dst_emb, src_neg_emb, dst_neg_emb)
            loss_train = loss_function_entropysample(pos_logit, neg_logit, b_xent, loss_type=args.loss_type)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            epoch_loss += loss_train.item()
            print('--> Epoch %d Step %5d loss: %.3f' % (epoch + 1, steps + 1, loss_train.item()))
            steps += 1

        auc_, ap_ = get_roc_scoreGNN(model, features, adj, dataloader_val, device)
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
    auc_test, ap_test = get_roc_scoreGNN(model, features, adj, dataloader_test, device)
    return auc_test, ap_test


def finetune_GCLWeightGNN(features, device, args):
    print('---> Loading dataset for gcn...')
    if args.dataset == 'BlogCatalog' or args.dataset == 'ACM' or args.dataset == 'Flickr':
        adj, _, adj_org, labels, idx_train, idx_val, idx_test = load_citationANEmat(args.dataset)
    else:
        adj, _, labels, idx_train, idx_val, idx_test = load_citation(args.dataset)
    print('--->Generate train/valid links for unsupervised learning...')
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges_net_lp(
        adj)
    adj = adj_train

    features = sparse_mx_to_torch_sparse_tensor(features).float().to_dense()
    adj_norm = normalize(adj)

    adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm)

    dataloader = EdgeSampler(train_edges, train_edges_false, args.batch_size)
    dataloader_val = EdgeSampler(val_edges, np.array(val_edges_false), args.batch_size, remain_delet=False)
    dataloader_test = EdgeSampler(test_edges, np.array(test_edges_false), args.batch_size, remain_delet=False)
    pos_weight = torch.Tensor([float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()])
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    save_path = "./weights/LP_%s_" % args.model_type + args.dataset + '_%d_' % args.dim + '_%d_' % args.hid1\
                + '_%d_' % args.hid2 + '{}'.format(args.trade_weight) + '.pth'
    auc_test, ap_test = train_GCLGNN(features, adj_norm, dataloader, dataloader_val, dataloader_test, save_path, device, args,
                              pos_weight, norm)
    return auc_test, ap_test