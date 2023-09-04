import os
import sys
import random
import json
import argparse

import numpy as np
import pandas as pd

from functools import partial

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.utils.data.sampler import WeightedRandomSampler

from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from accelerate import Accelerator

import db_utils
import model_utils
import eval_tools
from torchsampler import ImbalancedDatasetSampler
from generate_db import get_all_combinations

from scipy.spatial.distance import cdist

from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)


def main(
        id,
        data_prop,
        label_prop,
        objective,
        lr,
        bs,
        moa,
        target,
        layers,
        width,
        batch_effect_correct,

        # Cellprofiler is the baseline model
        cell_profiler=False,

        # Choose to eval a pretrained model or not
        ckpt=None,

        # These are not used but come from DB
        reserved=None,
        finished=None,

        # Defaults below are fixed
        epochs=2000,  # 500,
        warmup_steps=100,  # 50,
        warmup_epochs=5,
        early_stopping=True,
        stop_criterion=10,  # 16,
        test_epochs=500,
        version=24,
        inchi_key="ZWYQUVBAZLHTHP-UHFFFAOYSA-N",
        final_data="/media/data_cifs/projects/prj_video_imagenet/sys_linear_models/assay_data.npz",
        perturb_data="/media/data/final_data.npz",
        ckpt_dir="/media/data/sys_ckpts",
    ):
    """Run one iteration of training and evaluation."""
    accelerator = Accelerator()
    device = accelerator.device
    # device = "cuda"

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
    out_name = "data_{}_model_{}_{}_{}_{}_{}.pth".format(version, id, data_prop, label_prop, width, layers)
    path = os.path.join(ckpt_dir, out_name)

    # Load data
    data = np.load(final_data, allow_pickle=True)
    train_res, train_source, train_batch, train_well, train_compounds = data["train_res"], data["train_source"], data["train_batch"], data["train_well"], data["train_compounds"]
    test_res, test_source, test_batch, test_well, test_compounds = data["test_res"], data["test_source"], data["test_batch"], data["test_well"], data["test_compounds"]
    keys, compounds = data["keys"], data["compounds"]
    sel_keys = keys == inchi_key
    sel_comps = compounds[sel_keys]
    sel_train = train_compounds == np.unique(sel_comps).squeeze()
    sel_test = test_compounds == np.unique(sel_comps).squeeze()

    data = np.load(perturb_data)
    orf_data, orf_source, orf_batch, orf_well = data["orf_data"], data["orf_source"], data["orf_batch"], data["orf_well"]
    crispr_data, crispr_source, crispr_batch, crispr_well = data["crispr_data"], data["crispr_source"], data["crispr_batch"], data["crispr_well"]
    orfs, crisprs = data["orfs"], data["crisprs"]

    # Handle label and data prop at the same time
    # test_compounds, test_res, test_source, test_batch, test_well = model_utils.prepare_labels(test_compounds, test_res, test_source, test_batch, test_well, objective, label_prop, reference=train_compounds)

    # # Handle data prop - Always use full test set
    # train_compounds, train_res, train_source, train_batch, train_well = model_utils.prepare_data(train_compounds, train_res, train_source, train_batch, train_well, data_prop)
    bs = min(len(train_compounds), bs)  # Adjust so we dont overestimate batch

    # Counting variables for training
    best_loss = 10000000
    epoch_counter = 0
    balanced_loss = False
    best_test_acc = 0.
    # eb = None
    teb = None
    tops = 10  # top-K classification accuracy
    nc = len(np.unique(train_compounds))

    exps = {
        "data_prop": [0.01, .25, 0.5, .75, 1.],  # [0.1, 0.2, 0.4, 0.6, 0.8, 1.],  # np.arange(0, 1.1, 0.1),
        "label_prop": [0.01, .25, 0.5, .75, 1.],  # [0.1, 0.2, 0.4, 0.6, 0.8, 1.],  # np.arange(0, 1.1, 0.1),  # Proportion of labels, i.e. x% of molecules for labels
        "objective": ["mol_class"],  # "masked_recon"],
        "lr": [1e-4],
        "bs": [6000],
        "moa": [True],
        "target": [True],
        "layers": [1],
        "width": [1],
        "batch_effect_correct": [True],  # , False],
    }
    combos = get_all_combinations(exps)
    for idx, comb in tqdm(enumerate(combos), total=len(combos), desc="Counting data"):
        # Build model etc
        ic, _, _, _, _, _ = model_utils.prepare_labels(train_compounds, train_res, train_source, train_batch, train_well, sel_train, objective, label_prop=comb["label_prop"], data_prop=comb["data_prop"], reference=test_compounds)
        num = len(ic)
        combos[idx]["total_data"] = num

    # Make a dataframe with combos and params
    df = pd.DataFrame.from_dict(combos)
    df.to_csv("data_counts.csv")

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and test one model.")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--ckpt', type=str, default=None)
    args = parser.parse_args()

    if 1:
        # Run in debug mode
        params = {
            "id": -1,
            "data_prop": 1.,
            "label_prop": 1.,
            "objective": "mol_class",
            "lr": 1e-4,
            "bs": [6000][0],
            "moa": [True][0],
            "target": [True][0],
            "layers": 12,
            "width": 1512,
            "batch_effect_correct": [True, False][0],
            "cell_profiler": False
        }
    main(**params, ckpt=args.ckpt)

