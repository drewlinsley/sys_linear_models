"""Model utilities for large-scale assays."""
import torch
from torch import nn
import numpy as np


def get_obj_fun(objective):
    """Selector for objective functions."""
    if objective == "mol_class":
        obj = nn.CrossEntropyLoss
    elif objective == "masked_recon":
        obj = nn.MSELoss  # masking_loss
    elif obj == "barlow":
        obj = barlow_loss
    else:
        raise NotImplementedError(objective)
    return obj


class barlow_loss:
    def __init__():
        raise NotImplementedError
        pass

    def __call__(X, y=None):
        # Compare X to y with L2 loss
        return nn.MSELoss()(X, y)


def preprocess(x, y, objective, label_prop):
    if objective == "masked_recon":
        y = x.clone()
        mask = torch.rand_like(x, requires_grad=False) < label_prop
        x[mask] = -100
    return x, y


def prepare_labels(y, x, s, b, w, objective, label_prop, data_prop, reference=None):
    """Control how many labels vs. data you keep for training."""
    # Keep labels overlapping with reference
    assert reference is not None, "Pass test set as reference"
    h = {k: False for k in np.unique(reference)}
    main_ids = []
    for idx, e in enumerate(y):
        if e in h and not h[e]:
            main_ids.append(idx)
            h[e] = True
    main_ids = np.asarray(main_ids)  # 1 exemplar per test category. Minimal set
    # main_mask = np.in1d(y, np.unique(reference))
    # main_ids = np.where(main_mask)[0]
    # additional_mask = np.where(~main_mask)[0]
    additional_mask = np.arange(len(y))[~np.in1d(np.arange(len(y)), main_ids)]
    if objective == "mol_class" and label_prop < 1:
        # First figure out how many additional compounds/labels to keep
        ul = np.unique(y[additional_mask])
        keep_labels = int(len(ul) * label_prop)
        ul = ul[:keep_labels]
        additional_mask = additional_mask[np.in1d(y[additional_mask], ul)]

    # Next figure out what proportion of data to keep
    keep_count = int(len(additional_mask) * data_prop)
    additional_mask = additional_mask[:keep_count]
    keep = np.concatenate((main_ids, additional_mask))
    y = y[keep]
    x = x[keep]
    s = s[keep]
    b = b[keep]
    w = w[keep]
    return y, x, s, b, w


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

