import torch
import torch.nn.modules.loss
import torch.nn.functional as F


def loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD


def loss_function_entropy(preds, labels, norm, pos_weight):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)
    return cost


def loss_function_entropysample(pos_logit, neg_logit, b_xent, loss_type='entropy'):
    pos_logit = pos_logit.view(-1, 1)
    neg_logit = neg_logit.view(pos_logit.shape[0], -1)
    if loss_type == 'entropy':
        logits = torch.cat([pos_logit, neg_logit], dim=1)
        labels = torch.cat([torch.ones_like(pos_logit), torch.zeros_like(neg_logit)], dim=1)
        cost = b_xent(logits, labels)
    else:
        cost = torch.mean(torch.nn.functional.softplus(neg_logit - pos_logit))
    return cost


def cross_entropy_loss(preds, labels):
    cost = F.binary_cross_entropy_with_logits(preds, labels)
    return cost
