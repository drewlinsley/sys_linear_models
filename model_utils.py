"""Model utilities for large-scale assays."""
import torch
from torch import nn
import numpy as np


def get_obj_fun(objective):
    """Selector for objective functions."""
    if objective == "mol_class":
        obj = CCE  # nn.CrossEntropyLoss
    elif objective == "masked_recon":
        obj = MSE  # nn.MSELoss  # masking_loss
    elif objective == "denoising":
        obj = MSE  # nn.MSELoss  # masking_loss
    elif obj == "barlow":
        obj = barlow_loss
    else:
        raise NotImplementedError(objective)
    return obj


class CCE:
    def __init__(self):
        pass

    def __call__(self, X, y, mask):
        # Compare X to y with L2 loss
        return nn.CrossEntropyLoss()(X, y)


class MSE:
    def __init__(self):
        pass

    def __call__(self, X, y, mask):
        # Compare X to y with L2 loss
        X = X[mask]
        y = y[mask]
        return nn.MSELoss()(X, y)


class barlow_loss:
    def __init__(lambd=0.0051):
        self.lambd = lambd

    def __call__(X, y=None):
        # Compare X to y with L2 loss
        # empirical cross-correlation matrix

        # sum the cross-correlation matrix between all gpus
        X.div_(len(X))
        # torch.distributed.all_reduce(X)

        on_diag = torch.diagonal(X).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(X).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss


def preprocess(x, y, objective, label_prop):
    if objective == "masked_recon":
        y = x.clone()
        mask = torch.rand_like(x, requires_grad=False) < label_prop
        x[mask] = -100
    elif objective == "denoising":
        y = x.clone()
        x = x  #  + torch.rand_like(x) * 0.01
        mask = 0
    else:
        mask = 0
    return x, y, mask


def prepare_labels(y, x, s, b, w, i, objective, label_prop, data_prop, reference=None):
    """Control how many labels vs. data you keep for training."""
    # Keep labels overlapping with reference
    assert reference is not None, "Pass test set as reference"
    # h = {k: False for k in np.unique(reference)}
    # main_ids = []
    # for idx, e in enumerate(y):
    #     if e in h:  #  and not h[e]:
    #         main_ids.append(idx)
    #         h[e] = True
    # main_ids = np.asarray(main_ids)  # 1 exemplar per test category. Minimal set
    # additional_mask = np.where(~main_mask)[0]
    if objective == "mol_class" and label_prop < 1:
        main_mask = np.in1d(y, np.unique(reference))
        main_ids = np.where(main_mask)[0]
        additional_mask = np.arange(len(y))[~np.in1d(np.arange(len(y)), main_ids)]
        # First figure out how many additional compounds/labels to keep
        ul = np.unique(y[additional_mask])
        keep_labels = int(len(ul) * label_prop)
        ul = ul[:keep_labels]
        additional_mask = additional_mask[np.in1d(y[additional_mask], ul)]

        # Update after label control
        keep = np.concatenate((main_ids, additional_mask))
        y = y[keep]
        x = x[keep]
        s = s[keep]
        b = b[keep]
        w = w[keep]
        i = i[keep]

    if data_prop < 1:
        uc = np.unique(y)
        keep_counts = {}
        for c in uc:
            thresh = np.ceil((y == c).sum() * data_prop)
            keep_counts[c] = thresh

        keep_h = {k: 0 for k in uc}
        keep = []
        for idx, c in enumerate(y):
            if keep_h[c] <= keep_counts[c]:
                keep.append(idx)
                keep_h[c] += 1
        keep = np.asarray(keep)

        # Get appropriate indices
        y = y[keep]
        x = x[keep]
        s = s[keep]
        b = b[keep]
        w = w[keep]
        i = i[keep]

    # # Next figure out what proportion of data to keep
    # uc = y[keep]
    # keep_counts = {}

    # keep_count = int(len(additional_mask) * data_prop)
    # additional_mask = additional_mask[:keep_count]
    # keep = np.concatenate((main_ids, additional_mask))

    # # Add on additional IDs for embedding test
    # embs = np.where(i)[0]
    # for e in embs:
    #     if e not in keep:
    #         keep = np.concatenate((keep, [e]))
    # # keep = np.concatenate((keep, np.where(i)[0]))

    # # Get appropriate indices
    # y = y[keep]
    # x = x[keep]
    # s = s[keep]
    # b = b[keep]
    # w = w[keep]
    # i = i[keep]
    return y, x, s, b, w, i


def prepare_data(y, x, s, b, w, data_prop):
    mask = np.zeros(len(y))
    l = int(data_prop * len(y))
    mask[:l] = 1
    mask = mask == 1
    y = y[mask]
    x = x[mask]
    s = s[mask]
    b = b[mask]
    w = w[mask]
    return y, x, s, b, w

