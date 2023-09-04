import os
import sys
import random
import json
import argparse

import numpy as np
import pandas as pd

from functools import partial
from typing import Tuple, Optional, Union

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

from scipy.spatial.distance import cdist

from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)


class Transformer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    # def forward(self, x, y=None, mask=None):
    def forward(self, dv, iv_s, iv_b, iv_w, return_p = False, y=None, mask=None):
        x = dv[:, None]
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec: # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        out = self.out(x.squeeze(1))
        b, s, w = 0, 0, 0
        if return_p:
            return out, x
        else:
            return out, b, s, w


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
                 embedding_dim_w,
                 batch_effect_correct,
                 objective,
                 norm_type=nn.BatchNorm1d,  # nn.LayerNorm,  # torch.nn.Identity,  # torch.nn.BatchNorm1d,
                 use_dropout=0.,
                 num_heads=2,
                 num_layers=9,
                 dim_ref=None,
                 mlp_ratio=2,
                 act=lambda x: F.gelu(x, approximate="tanh"),
                 norm_layer: nn.Module = nn.LayerNorm):
        super(Transformer, self).__init__()
        dim_self = 878
        self.enc_dec = False
        dim_ref = dim_ref if dim_ref is not None else dim_self
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and self.enc_dec:  # cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            elif self.enc_dec:  # self
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:  # self or cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)
        self.out = torch.nn.Linear(dim_ref, output_dim)


class TransformerLayer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=lambda x: F.gelu(x, approximate="tanh"),
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)


class MultiHeadAttention(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        # self.q_norm = nn.LayerNorm(dim_self, elementwise_affine=True)
        # self.k_norm = nn.LayerNorm(dim_self * 2, elementwise_affine=True)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # queries = self.q_norm(self.to_queries(x)).reshape(b, n, self.num_heads, c // self.num_heads)
        # # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        # keys_values = self.k_norm(self.to_keys_values(y)).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention


class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.GELU):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
                layers.append(nn.LayerNorm(sizes[i + 1], elementwise_affine=True))
        self.model = nn.Sequential(*layers)


class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=lambda x: F.gelu(x, approximate="tanh"), dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


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
                 embedding_dim_w,
                 batch_effect_correct,
                 objective,
                 norm_type=nn.BatchNorm1d,  # nn.LayerNorm,  # torch.nn.Identity,  # torch.nn.BatchNorm1d,
                 use_dropout=0.):
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
                norm_type(embedding_dim_s)
            ])
        self.proj = nn.Sequential(*[
            nn.Linear(input_dim, int_dim),
            norm_type(int_dim)
        ])

        self.iv_layers, self.dv_layers = [], []
        for l in range(self.n_blocks):
            if self.batch_effect_correct:
                self.dv_layers.append(nn.Sequential(*[
                    torch.nn.Linear(int_dim + embedding_dim_s, int_dim),
                    torch.nn.Dropout(use_dropout),
                    # norm_type(int_dim),  # BatchNorm1d(dim),
                    torch.nn.GELU(),
                    norm_type(int_dim),  # BatchNorm1d(dim),
                ]))
            else:
                self.dv_layers.append(nn.Sequential(*[
                    torch.nn.Linear(int_dim, int_dim),
                    torch.nn.Dropout(use_dropout),
                    # norm_type(int_dim),  # BatchNorm1d(dim),
                    torch.nn.GELU(),
                    norm_type(int_dim),  # BatchNorm1d(dim),
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
            x_s = self.embedding_s(iv_s).squeeze(1)
        for l in range(self.n_blocks):
            dv_layer = self.dv_layers[l]
            if self.batch_effect_correct:
                cat_y = torch.concat((y, x_s), 1)
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
        warmup_steps=50,
        warmup_epochs=50,
        early_stopping=True,
        stop_criterion=20,  # 16,
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
    emb_dim_b = 64
    emb_dim_s = 16  # 4
    emb_dim_p = 16
    emb_dim_w = 16
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

    # # Inverse weighting for sampling
    # uni_c, class_sample_count = np.unique(train_compounds, return_counts=True)
    # weight = 1. / class_sample_count
    # weight_dict = {k: v for k, v in zip(uni_c, weight)}
    # samples_weight = np.array([weight_dict[t] for t in train_compounds])
    # samples_weight = torch.from_numpy(samples_weight)
    # sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

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
    sampler = ImbalancedDatasetSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        drop_last=True,
        batch_size=bs,
        sampler=sampler,
        pin_memory=True)  # Remove pin memory if using accelerate
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=bs,
        drop_last=False,
        shuffle=False,
        pin_memory=False)

    # Build model etc
    model = Transformer(
        input_dim,
        width,
        output_dim,
        layers,
        objective=objective,
        batch_effect_correct=batch_effect_correct,
        num_embeddings_b=num_embeddings_b,
        num_embeddings_s=num_embeddings_s,
        num_embeddings_w=num_embeddings_w,
        embedding_dim_b=emb_dim_b,
        embedding_dim_s=emb_dim_s,
        embedding_dim_w=emb_dim_w)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        weight_decay=1e-4,  # Default
        lr=lr)  # ,
    scheduler = get_cosine_schedule_with_warmup(  # get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=epochs * int(len(train_loader) // bs)
    )

    # Objective function
    obj_fun = model_utils.get_obj_fun(objective)
    model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        test_loader,
        scheduler)
    # model.to(device)
    avg_loss = torch.tensor(0).float().to(device)

    if cell_profiler:
        # Pass orginal normalized encodings
        train_enc = train_res
        train_lab = train_compounds
        test_enc = test_res
        test_lab = test_compounds
        best_loss = -1  # Dummy
        width = train_res.shape[1]
        print("Skipping training — using cell profiler instead.")
    else:
        if ckpt is not None:
            path = ckpt
        else:
            # accelerator.wait_for_everyone()
            for epoch in range(epochs):
                batch_losses = []
                # progress = tqdm(total=len(train_loader), desc="Training", disable=not accelerator.is_local_main_process)
                progress = tqdm(total=len(train_loader), desc="Training")
                model.train()
                for batch_idx, batch in enumerate(train_loader):  # tqdm(enumerate(sample1_loader), total=len(sample1_loader), desc="Epoch"):            
                    optimizer.zero_grad(set_to_none=True)
                    dv, text_embeddings, iv_s, iv_b, iv_w = batch

                    # Preprocess data for some model approaches
                    dv, text_embeddings, mask = model_utils.preprocess(dv, text_embeddings, objective, label_prop)

                    # Move data to GPU. Only needed when we dont use accelerate
                    # dv = dv.to(device)
                    # text_embeddings = text_embeddings.to(device)
                    # iv_s = iv_s.to(device)
                    # iv_b = iv_b.to(device)
                    # iv_w = iv_w.to(device)
                    #
                    image_embeddings, b, s, w = model(dv=dv, iv_s=iv_s, iv_b=iv_b, iv_w=iv_w)

                    # Make entropic targets
                    loss = obj_fun()(image_embeddings, text_embeddings, mask)
                    if batch_effect_correct:  # eb is None and batch_effect_correct:
                        eb = F.softmax(torch.ones_like(b), 1)
                        es = F.softmax(torch.ones_like(s), 1)
                        ew = F.softmax(torch.ones_like(w), 1)
                        bl = F.cross_entropy(b, eb)
                        sl = F.cross_entropy(s, es)
                        wl = F.cross_entropy(w, ew)
                        loss = loss + bl + sl + wl

                    # Optimize
                    accelerator.backward(loss)
                    # loss.backward()
                    optimizer.step()
                    scheduler.step()  # Lets go without a scheduler for now
                    batch_losses.append(loss)
                    progress.set_postfix({"train_loss": loss})  # , "compounds": comp_loss, "phenotypes": pheno_loss})
                    progress.update()

                # Run test set
                test_losses, test_accs = [], []
                model.eval()
                with torch.no_grad():
                    for batch_idx, batch in enumerate(test_loader):  # tqdm(enumerate(sample1_loader), total=len(sample1_loader), desc="Epoch"):
                        dv, text_embeddings, iv_s, iv_b, iv_w = batch

                        # Preprocess data for some model approaches
                        dv, text_embeddings, mask = model_utils.preprocess(dv, text_embeddings, objective, label_prop)

                        # Move data to GPU. Only needed when we dont use accelerate
                        # dv = dv.to(device)
                        # text_embeddings = text_embeddings.to(device)
                        # iv_b = iv_b.to(device)
                        # iv_s = iv_s.to(device)
                        # iv_w = iv_w.to(device)
                        #

                        image_embeddings, b, s, w = model(dv=dv, iv_s=iv_s, iv_b=iv_b, iv_w=iv_w)
                        # image_embeddings = model(dv=dv, iv_s=iv_s, iv_b=iv_b, iv_w=iv_w)
                        loss = obj_fun()(image_embeddings, text_embeddings, mask)
                        # Losses to become invariant to batch effects
                        # _, tk = torch.topk(image_embeddings, tops, dim=1)
                        # accuracy = (tk == text_embeddings[:, None]).sum(1).float().sum() / len(tk)
                        test_losses.append(loss)
                        # test_accs.append(accuracy)

                # Check performances
                epoch_loss = np.mean([x.item() for x in batch_losses])
                test_loss = np.mean([x.item() for x in test_losses])
                test_acc = np.mean([x.item() for x in test_accs]) * 100.
                if 1:  # accelerator.is_main_process:
                    if test_loss < best_loss:
                        print("Saving best performing weights")
                        best_loss = test_loss
                        # best_test_acc = test_acc
                        torch.save(model.state_dict(), path)
                        epoch_counter = 0
                    else:
                        if epoch > warmup_epochs:  # Start checking for early stopping
                            epoch_counter += 1
                    # progress.set_postfix({"epoch": epoch, "number_compounds": nc, "train_loss": epoch_loss, "test_loss": test_loss, "test_acc": test_acc, "best_test_acc": best_test_acc, "well_loss": wl, "batch_loss": bl, "source_loss": sl})
                    progress.set_postfix({"epoch": epoch, "number_compounds": nc, "train_loss": epoch_loss, "test_loss": test_loss, "best_loss": best_loss})
                    progress.update()
                progress.close()
                # accelerator.wait_for_everyone()

                # Handle early stopping
                if epoch_counter > stop_criterion:
                    print("Triggered early stopping.")
                    break  # Early stopping is initiated
            print('Finished training')

        # Load best weights
        model.load_state_dict(torch.load(path))
        model.eval()

        # Encode training set
        print('Begin evaluation')
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=bs * 2,
            drop_last=False,
            shuffle=False)
        # train_loader = accelerator.prepare(train_loader)
        train_enc, train_lab = [], []
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc="Processing training data"):
                dv, text_embeddings, iv_s, iv_b, iv_w = batch

                # # Preprocess data for some model approaches
                # dv, text_embeddings, mask = model_utils.preprocess(dv, text_embeddings, objective, label_prop)

                # Move data to GPU. Only needed when we dont use accelerate
                dv = dv.to(device)
                text_embeddings = text_embeddings.to(device)
                iv_b = iv_b.to(device)
                iv_s = iv_s.to(device)
                iv_w = iv_w.to(device)

                _, image_embeddings = model(dv=dv, iv_s=iv_s, iv_b=iv_b, iv_w=iv_w, return_p=True)
                train_enc.append(image_embeddings)
                train_lab.append(text_embeddings)
        train_enc = torch.cat(train_enc).cpu().numpy()
        train_lab = torch.cat(train_lab).cpu().numpy()

        # Encode test set
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=bs * 2,
            drop_last=False,
            shuffle=False)
        # test_loader = accelerator.prepare(test_loader)
        test_enc, test_lab = [], []
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc="Processing testing data"):
                dv, text_embeddings, iv_s, iv_b, iv_w = batch

                # # Preprocess data for some model approaches
                # dv, text_embeddings, mask = model_utils.preprocess(dv, text_embeddings, objective, label_prop)

                # Move data to GPU. Only needed when we dont use accelerate
                dv = dv.to(device)
                text_embeddings = text_embeddings.to(device)
                iv_b = iv_b.to(device)
                iv_s = iv_s.to(device)
                iv_w = iv_w.to(device)

                _, image_embeddings = model(dv=dv, iv_s=iv_s, iv_b=iv_b, iv_w=iv_w, return_p=True)
                test_enc.append(image_embeddings)
                test_lab.append(text_embeddings)
        test_enc = torch.cat(test_enc).cpu().numpy()  # Slow but convenient. Consider removing.
        test_lab = torch.cat(test_lab).cpu().numpy()

    # Encode ORFs and CRISPRs
    pos_orf_idx = orfs == "eGFP"
    neg_orf_idx = orfs == "LacZ"
    pos_crispr_idx = crisprs == "PLK1"
    neg_crispr_idx = crisprs == "non-targeting"
    pos_orf = orf_data[pos_orf_idx]
    neg_orf = orf_data[neg_orf_idx]
    pos_crispr = crispr_data[pos_crispr_idx]
    neg_crispr = crispr_data[neg_crispr_idx]
    pos_orf_batch, pos_orf_source, pos_orf_well = orf_batch[pos_orf_idx], orf_source[pos_orf_idx], orf_well[pos_orf_idx]
    neg_orf_batch, neg_orf_source, neg_orf_well = orf_batch[neg_orf_idx], orf_source[neg_orf_idx], orf_well[neg_orf_idx]
    pos_crispr_batch, pos_crispr_source, pos_crispr_well = crispr_batch[pos_crispr_idx], crispr_source[pos_crispr_idx], crispr_well[pos_crispr_idx]
    neg_crispr_batch, neg_crispr_source, neg_crispr_well = crispr_batch[neg_crispr_idx], crispr_source[neg_crispr_idx], crispr_well[neg_crispr_idx]
    if cell_profiler:
        pos_orf_emb = pos_orf
        neg_orf_emb = neg_orf
        pos_crispr_emb = pos_crispr
        neg_crispr_emb = neg_crispr
    else:
        with torch.no_grad():
            pos_orf, neg_orf, pos_orf_batch, pos_orf_source, pos_orf_well, neg_orf_batch, neg_orf_source, neg_orf_well = torch.from_numpy(pos_orf).to(device), torch.from_numpy(neg_orf).to(device), torch.from_numpy(pos_orf_batch).to(device), torch.from_numpy(pos_orf_source).to(device), torch.from_numpy(pos_orf_well).to(device), torch.from_numpy(neg_orf_batch).to(device), torch.from_numpy(neg_orf_source).to(device), torch.from_numpy(neg_orf_well).to(device)
            pos_crispr, neg_crispr, pos_crispr_batch, pos_crispr_source, pos_crispr_well, neg_crispr_batch, neg_crispr_source, neg_crispr_well = torch.from_numpy(pos_crispr).to(device), torch.from_numpy(neg_crispr).to(device), torch.from_numpy(pos_crispr_batch).to(device), torch.from_numpy(pos_crispr_source).to(device), torch.from_numpy(pos_crispr_well).to(device), torch.from_numpy(neg_crispr_batch).to(device), torch.from_numpy(neg_crispr_source).to(device), torch.from_numpy(neg_crispr_well).to(device)

            # Get embs
            _, pos_orf_emb = model(dv=pos_orf, iv_s=pos_orf_source, iv_b=pos_orf_batch, iv_w=pos_orf_well, return_p=True)
            _, neg_orf_emb = model(dv=neg_orf, iv_s=neg_orf_source, iv_b=neg_orf_batch, iv_w=neg_orf_well, return_p=True)
            _, pos_crispr_emb = model(dv=pos_crispr, iv_s=pos_crispr_source, iv_b=pos_crispr_batch, iv_w=pos_crispr_well, return_p=True)
            _, neg_crispr_emb = model(dv=neg_crispr, iv_s=neg_crispr_source, iv_b=neg_crispr_batch, iv_w=neg_crispr_well, return_p=True)
            pos_orf_emb, neg_orf_emb = pos_orf_emb.cpu(), neg_orf_emb.cpu()
            pos_crispr_emb, neg_crispr_emb = pos_crispr_emb.cpu(), neg_crispr_emb.cpu()

    # Compute z primes (Actually just a measure of separability...)
    orf_ds = cdist(pos_orf_emb, neg_orf_emb, metric="euclidean").mean(1)
    crispr_ds = cdist(pos_crispr_emb, neg_crispr_emb, metric="euclidean").mean(1)
    z_prime_orf = orf_ds.mean() / orf_ds.std()
    z_prime_crispr = crispr_ds.mean() / crispr_ds.std()

    # Run MoA test
    moa_perf = eval_tools.run(kind="moa", train_X=train_enc, train_y=train_lab, test_X=test_enc, test_y=test_lab, device=device, epochs=test_epochs, width=width, sel_train=sel_train, sel_test=sel_test)

    # Run Target test
    target_perf = eval_tools.run(kind="target", train_X=train_enc, train_y=train_lab, test_X=test_enc, test_y=test_lab, device=device, epochs=test_epochs, width=width)

    # Update the DB with results
    results = {
        "meta_id": id,
        "task_loss": best_loss,
        "moa_acc": moa_perf["acc"],
        "moa_loss": moa_perf["loss"],
        "moa_acc_std": moa_perf["acc_std"],
        "moa_loss_std": moa_perf["loss_std"],
        "target_acc": target_perf["acc"],
        "target_loss": target_perf["loss"],
        "target_acc_std": target_perf["acc_std"],
        "target_loss_std": target_perf["loss_std"],
        "rediscovery_acc": moa_perf["rediscovery_acc"],
        "rediscovery_z": moa_perf["rediscovery_z"],
        "z_prime_orf": z_prime_orf,
        "z_prime_crispr": z_prime_crispr
    }
    db_utils.record_performance(results)

    # Clean up weights
    if not cell_profiler:
        os.remove(path)
    print(json.dumps(results, indent=2))
    print("Finished")
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and test one model.")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--ckpt', type=str, default=None)
    args = parser.parse_args()

    if args.debug:
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
            "layers": 6,
            "width": 1024,
            "batch_effect_correct": [True, False][1],
            "cell_profiler": False
        }
    else:
        # Run in DB mode
        params = db_utils.get_and_reserve_params()
    print(json.dumps(params, indent=2))
    main(**params, ckpt=args.ckpt)

