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


class Mmd_resnet(nn.Module):   
    def __init__(self, 
                 input_dim,
                 int_dim,
                 output_dim,
                 n_blocks,
                 num_embeddings_b,
                 num_embeddings_s,
                 # num_embeddings_p,
                 num_embeddings_w,
                 embedding_dim_b,
                 embedding_dim_s,
                 # embedding_dim_p,
                 # embedding_dim_w,
                 batch_effect_correct,
                 objective,
                 norm_type=nn.BatchNorm1d,  # nn.LayerNorm,  # torch.nn.Identity,  # torch.nn.BatchNorm1d,
                 use_dropout=0.1):
        super(Mmd_resnet, self).__init__()    

        self.n_blocks = n_blocks
        self.batch_effect_correct = batch_effect_correct

        # Make IV and DV networks
        if self.batch_effect_correct:
            self.embedding_s = nn.Sequential(*[
                torch.nn.Embedding(
                    num_embeddings=num_embeddings_s,
                    embedding_dim=embedding_dim_s),
                torch.nn.Linear(embedding_dim_s, embedding_dim_s),
                norm_type(embedding_dim_s),
                torch.nn.GELU(),
                # norm_type(embedding_dim_s)
            ])
            self.embedding_b = nn.Sequential(*[
                torch.nn.Embedding(
                    num_embeddings=num_embeddings_b,
                    embedding_dim=embedding_dim_b),
                torch.nn.Linear(embedding_dim_b, embedding_dim_b),
                norm_type(embedding_dim_b),
                torch.nn.GELU(),
                # norm_type(embedding_dim_s)
            ])
            self.s_layers = []
            for l in range(self.n_blocks):
                self.s_layers.append(nn.Sequential(*[
                    torch.nn.Linear(embedding_dim_s + embedding_dim_b, embedding_dim_s + embedding_dim_b),
                    # torch.nn.Dropout(use_dropout),
                    # torch.nn.GELU(),
                    norm_type(embedding_dim_s + embedding_dim_b),
                    torch.nn.GELU(),
                ]))
            self.s_layers = nn.ModuleList(self.s_layers)

        self.proj = nn.Sequential(*[
            nn.Linear(input_dim, int_dim),
            norm_type(int_dim)
        ])

        self.iv_layers, self.dv_layers = [], []
        for l in range(self.n_blocks):
            if self.batch_effect_correct:
                self.dv_layers.append(nn.Sequential(*[
                    torch.nn.Linear(int_dim + embedding_dim_s + embedding_dim_b, int_dim),
                    norm_type(int_dim),  # BatchNorm1d(dim),
                    torch.nn.Dropout(use_dropout),
                    torch.nn.GELU(),
                    # norm_type(int_dim),  # BatchNorm1d(dim),
                ]))
            else:
                self.dv_layers.append(nn.Sequential(*[
                    torch.nn.Linear(int_dim, int_dim),
                    norm_type(int_dim),  # BatchNorm1d(dim),
                    torch.nn.Dropout(use_dropout),
                    torch.nn.GELU(),
                    # norm_type(int_dim),  # BatchNorm1d(dim),
                ]))
        self.dv_layers = nn.ModuleList(self.dv_layers)
        if objective == "barlow":
            self.final = nn.Sequential([
                torch.nn.Linear(int_dim, output_dim),
                norm_type(int_dim),
                torch.nn.GELU(),
                torch.nn.Linear(int_dim, output_dim)
            ])
            self.bn = nn.BatchNorm1d(output_dim, affine=False)
        elif objective == "denoising":
            self.final = nn.Sequential(*[
                torch.nn.Linear(int_dim, int_dim // 4),
                norm_type(int_dim // 4),
                torch.nn.GELU(),
                torch.nn.Linear(int_dim // 4, int_dim // 4 // 4),
                norm_type(int_dim // 4 // 4),
                torch.nn.GELU(),
                torch.nn.Linear(int_dim // 4 // 4, int_dim // 4),
                norm_type(int_dim // 4),
                torch.nn.GELU(),
                torch.nn.Linear(int_dim // 4, output_dim)
            ])
        else:
            self.final = nn.Sequential(*[
                torch.nn.Linear(int_dim, output_dim)
            ])
        if self.batch_effect_correct:
            self.b = nn.Linear(int_dim, num_embeddings_b)
            self.s = nn.Linear(int_dim, num_embeddings_s)
            self.w = nn.Linear(int_dim, num_embeddings_w)
        
    def forward(self, dv, iv_s, iv_b, iv_w, return_p = False):
        """Forward function (with skip connections)"""
        y = self.proj(dv)
        if self.batch_effect_correct:
            it_s = self.embedding_s(iv_s).squeeze(1)
            it_b = self.embedding_b(iv_b).squeeze(1)
            it_s = torch.cat((it_s, it_b), 1)
        for l in range(self.n_blocks):
            dv_layer = self.dv_layers[l]
            if self.batch_effect_correct:
                it_s = self.s_layers[l](it_s)
                # x_s = self.s_layers[l](x_s)
                # cat_y = torch.concat((y, x_s), 1)
                cat_y = torch.concat((y, it_s), 1)
            else:
                cat_y = y
            if l % 2:  # Skip
                y = dv_layer(cat_y) + y
            else:  # Dense
                y = dv_layer(cat_y)
        out = self.final(y)
        if self.batch_effect_correct:
            b = self.b(y)
            s = self.s(y)
            w = self.w(y)
        else:
            b, s, w = y, y, y
        if return_p:
            return out, y
        else:
            return out, b, s, w

     
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
    train_compounds, train_res, train_source, train_batch, train_well, sel_train = model_utils.prepare_labels(train_compounds, train_res, train_source, train_batch, train_well, sel_train, objective, label_prop=label_prop, data_prop=data_prop, reference=test_compounds)
    # test_compounds, test_res, test_source, test_batch, test_well = model_utils.prepare_labels(test_compounds, test_res, test_source, test_batch, test_well, objective, label_prop, reference=train_compounds)

    # # Handle data prop - Always use full test set
    # train_compounds, train_res, train_source, train_batch, train_well = model_utils.prepare_data(train_compounds, train_res, train_source, train_batch, train_well, data_prop)
    bs = min(len(train_compounds), bs)  # Adjust so we dont overestimate batch

    # Default params
    emb_dim_b = 16  # 64
    emb_dim_s = 16  # 4
    # emb_dim_p = 16  # 64  # 16
    # emb_dim_w = 16  # 128  # 16
    num_embeddings_b = train_batch.max() + 1  # train_batch.shape[-1]
    num_embeddings_s = train_source.max() + 1  # train_source.shape[-1]
    num_embeddings_w = train_well.max() + 1  # train_well.shape[-1]
    input_dim = train_res.shape[-1]
    if objective == "mol_class":
        output_dim = train_compounds.max() + 1
    elif objective == "masked_recon" or objective == "denoising":
        output_dim = train_res.shape[1]
    else:
        raise NotImplementedError(objective)

    # Counting variables for training
    best_loss = 10000000
    epoch_counter = 0
    balanced_loss = False
    best_test_acc = 0.
    # eb = None
    teb = None
    tops = 10  # top-K classification accuracy
    nc = len(np.unique(train_compounds))

    # Inverse weighting for sampling
    uni_c, class_sample_count = np.unique(train_compounds, return_counts=True)
    weight = 1. / class_sample_count
    weight_dict = {k: v for k, v in zip(uni_c, weight)}
    samples_weight = np.array([weight_dict[t] for t in train_compounds])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    # Build data
    train_res = torch.Tensor(train_res).float()
    train_source = torch.Tensor(train_source).long()
    train_batch = torch.Tensor(train_batch).long()
    train_well = torch.Tensor(train_well).long()
    train_compounds = torch.Tensor(train_compounds).long()
    test_res = torch.Tensor(test_res).float()
    test_source = torch.Tensor(test_source).long()
    test_batch = torch.Tensor(test_batch).long()
    test_well = torch.Tensor(test_well).long()
    test_compounds = torch.Tensor(test_compounds).long()
    train_dataset = torch.utils.data.TensorDataset(
        train_res,
        train_compounds,
        train_source,
        train_batch,
        train_well)
    test_dataset = torch.utils.data.TensorDataset(
        test_res,
        test_compounds,
        test_source,
        test_batch,
        test_well)
    # sampler = ImbalancedDatasetSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        drop_last=True,
        batch_size=bs,
        shuffle=True,
        # sampler=sampler,
        pin_memory=True)  # Remove pin memory if using accelerate
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=bs,
        drop_last=False,
        shuffle=False,
        pin_memory=False)

    exps = {
        "data_prop": [0.01],  # [0.1, 0.2, 0.4, 0.6, 0.8, 1.],  # np.arange(0, 1.1, 0.1),
        "label_prop": [0.01],  # [0.1, 0.2, 0.4, 0.6, 0.8, 1.],  # np.arange(0, 1.1, 0.1),  # Proportion of labels, i.e. x% of molecules for labels
        "objective": ["mol_class"],  # "masked_recon"],
        "lr": [1e-4],
        "bs": [6000],
        "moa": [True],
        "target": [True],
        "layers": [1, 3, 6, 9, 12],  # [1, 3, 6, 12],
        "width": [128, 256, 512, 1024, 1512],  # [256, 512, 768],  # [256, 512, 768, 1024],
        "batch_effect_correct": [True],  # , False],
    }
    combos = get_all_combinations(exps)

    counts = []
    for idx, comb in tqdm(enumerate(combos), total=len(combos)):
        # Build model etc
        model = Mmd_resnet(
            input_dim,
            comb["width"],
            output_dim,
            comb["layers"],
            objective=objective,
            batch_effect_correct=batch_effect_correct,
            num_embeddings_b=num_embeddings_b,
            num_embeddings_s=num_embeddings_s,
            num_embeddings_w=num_embeddings_w,
            embedding_dim_b=emb_dim_b,
            embedding_dim_s=emb_dim_s)
        num_parameters = sum(p.numel() for p in model.parameters())
        counts.append(num_parameters)
        combos[idx]["counts"] = num_parameters

        # Free up memory
        del model
        torch.cuda.empty_cache()

    # Make a dataframe with combos and params
    df = pd.DataFrame.from_dict(combos)
    df.to_csv("param_counts.csv")

    
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

