# -*- coding: utf-8 -*-
"""
Created on Sat Apr 6 00:17:31 2020

@author: urixs
"""

import numpy as np
from sklearn import decomposition
import argparse
from itertools import count
import os
from functools import partial
import pandas as pd

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import lr_scheduler
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)
from torch.utils.data.sampler import WeightedRandomSampler

from accelerate import Accelerator

import random
import string


# Returns a random alphanumeric string of length 'length'
def random_key(length):
    key = ''
    for i in range(length):
        key += random.choice(string.lowercase + string.uppercase + string.digits)
    return key


def renumber(x, return_dict=False):
    h = {}
    count = 0 
    for c in x:
        if c not in h:
            h[c] = count
            count += 1
    x = [h[c] for c in x]
    x = np.asarray(x)
    if return_dict:
        return x, h
    else:
        return x


def unique_map(x):
    # u, inv = np.unique(x, return_inverse=True)
    u = np.unique(x)
    h = {i: idx for idx, i in enumerate(u)}
    inv = [h[t] for t in x]
    return inv, h


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


# ==============================================================================
# =                                Input arguments                            =
# ==============================================================================

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--n_blocks', 
                    type=int, 
                    default=3, 
                    help='Number of resNet blocks')
parser.add_argument('--file1', 
                    type=str, 
                    default="./data/sample1.csv",
                    help='path to file 1')
parser.add_argument('--file2', 
                    type=str, 
                    default="./data/sample2.csv",
                    help='path to file 2')
parser.add_argument("--scale_k",
                    type=int,
                    default=5,
                    help="Number of neighbors for determining the RBF scale")
parser.add_argument('--batch_size', 
                    type=int, 
                    default=10000,
                    help='Batch size (default=128)')
parser.add_argument('--lr', 
                    type=float, 
                    default=1e-4,
                    help='learning_rate (default=1e-3)')
parser.add_argument("--min_lr",
                    type=float,
                    default=1e-6,
                    help="Minimal learning rate")
parser.add_argument("--decay_step_size",
                    type=int,
                    default=10,
                    help="LR decay step size")
parser.add_argument("--lr_decay_factor",
                    type=float,
                    default=0.1,
                    help="LR decay factor")
parser.add_argument("--weight_decay",
                    type=float,
                    default=1e-4,
                    help="l_2 weight penalty")
parser.add_argument("--epochs_wo_im",
                    type=int,
                    default=20,
                    help="Number of epochs without improvement before stopping")
parser.add_argument("--save_dir",
                    type=str,
                    default='./calibrated_data',
                    help="Directory for calibrated data")

args = parser.parse_args(args=[])


def aggregate_data(agg_fun, res, source, encodings, batch, plate, well, compounds):
    uc = np.unique(compounds)
    ares, asource, aencodings, abatch, aplate, awell, acompounds = [], [], [], [], [], [], []
    for u in uc:
        idx = u == compounds
        ares.append(agg_fun(res[idx]))
        asource.append(source[idx][0])
        aencodings.append(agg_fun(encodings[idx]))
        abatch.append(batch[idx][0])
        aplate.append(plate[idx][0])
        awell.append(well[idx][0])
        acompounds.append(compounds[idx][0])
    res = np.concatenate(ares, 0)
    source = np.asarray(asource)
    encodings = np.concatenate(aencodings)
    batch = np.asarray(abatch)
    well = np.asarray(awell)
    compounds = np.asarray(acompounds)
    return res, source, encodings, batch, plate, well, compounds


def unique_map(x):
    # u, inv = np.unique(x, return_inverse=True)
    u = np.unique(x)
    h = {i: idx for idx, i in enumerate(u)}
    inv = [h[t] for t in x]
    return inv, h


class ResnetBlock(nn.Module):
    """Define a Resnet block"""
    
    def __init__(self, 
                 dim,
                 use_dropout=False):
        """Initialize the Resnet block"""
        
        super(ResnetBlock, self).__init__()
        self.block = self.build_resnet_block(dim,
                                             use_dropout)

    def build_resnet_block(self,
                           dim,
                           use_dropout=False):
    
        block = [torch.nn.Linear(dim, dim),
                norm_type(dim),  # BatchNorm1d(dim),
                 torch.nn.SiLU()]
        if use_dropout:
            block += [nn.Dropout(0.5)]
        return nn.Sequential(*block)
    
    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.block(x)  # add skip connections
        return out

class MLP(nn.Module):
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
                 use_dropout=0.):
        super(MLP, self).__init__()

        self.n_blocks = n_blocks
        self.dv_layers = nn.Sequential(*[
            torch.nn.Linear(input_dim, int_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(use_dropout),
            torch.nn.Linear(int_dim, int_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(use_dropout),
            torch.nn.Linear(int_dim, int_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(use_dropout),
        ])
        self.final = nn.Sequential(*[
             # norm_type(int_dim),
             torch.nn.Linear(int_dim, output_dim)
        ])

    def forward(self, dv, iv_s, iv_b, iv_w):
        dv = self.dv_layers(dv)
        return self.final(dv)


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
                 norm_type=nn.BatchNorm1d,  # nn.LayerNorm,  # torch.nn.Identity,  # torch.nn.BatchNorm1d,
                 use_dropout=0.1):
        super(Mmd_resnet, self).__init__()    

        self.n_blocks = n_blocks

        # Make IV and DV networks
        # self.embedding_b = nn.Sequential(*[
        #     torch.nn.Embedding(
        #         num_embeddings=num_embeddings_b,
        #         embedding_dim=embedding_dim_b),
        #     torch.nn.Linear(embedding_dim_b, embedding_dim_b),
        #     norm_type(embedding_dim_b)
        # ])
        self.embedding_s = nn.Sequential(*[
            torch.nn.Embedding(
                num_embeddings=num_embeddings_s,
                embedding_dim=embedding_dim_s),
            torch.nn.Linear(embedding_dim_s, embedding_dim_s),
            norm_type(embedding_dim_s)
        ])
        # self.embedding_w = nn.Sequential(*[
        #     torch.nn.Embedding(
        #         num_embeddings=num_embeddings_w,
        #         embedding_dim=embedding_dim_w),
        #     torch.nn.Linear(embedding_dim_w, embedding_dim_w),
        #     norm_type(embedding_dim_w)
        # ])
        self.proj = nn.Sequential(*[
            nn.Linear(input_dim, int_dim),
            norm_type(int_dim)
        ])

        self.iv_layers, self.dv_layers = [], []
        for l in range(self.n_blocks):
            # self.iv_layers.append(nn.Sequential(*[
            #     torch.nn.Linear(embedding_dim, embedding_dim),
            #     torch.nn.SiLU(),
            #     norm_type(embedding_dim),  # BatchNorm1d(dim),
            # ]))
            self.dv_layers.append(nn.Sequential(*[
                # torch.nn.Linear(int_dim + embedding_dim_b + embedding_dim_s + embedding_dim_w, int_dim),
                torch.nn.Linear(int_dim + embedding_dim_s, int_dim),
                # torch.nn.Linear(int_dim + embedding_dim_s, int_dim),
                torch.nn.Dropout(use_dropout),
                torch.nn.GELU(),
                norm_type(int_dim),  # BatchNorm1d(dim),
            ]))
        # self.iv_layers = nn.ModuleList(self.iv_layers)
        self.dv_layers = nn.ModuleList(self.dv_layers)
        self.final = nn.Sequential(*[
            # norm_type(int_dim),
            torch.nn.Linear(int_dim, output_dim)
        ])
        self.b = nn.Linear(int_dim, num_embeddings_b)
        self.s = nn.Linear(int_dim, num_embeddings_s)
        self.w = nn.Linear(int_dim, num_embeddings_w)
        
    def forward(self, dv, iv_s, iv_b, iv_w, return_p = False):
        """Forward function (with skip connections)"""
        # out = input + self.model(input)  # add skip connection
        y = self.proj(dv)
        # x_b = self.embedding_b(iv_b).squeeze(1)
        x_s = self.embedding_s(iv_s).squeeze(1)
        # x_w = self.embedding_w(iv_w).squeeze(1)
        for l in range(self.n_blocks):
            # iv_layer, dv_layer = self.iv_layers[l], self.dv_layers[l]
            dv_layer = self.dv_layers[l]
            # x = iv_layer(x) + x
            # cat_y = torch.concat((y, x_b, x_s, x_w), 1)
            # cat_y = torch.concat((y, x_b, x_s), 1)
            cat_y = torch.concat((y, x_s), 1)
            if l % 2:
                y = dv_layer(cat_y) + y
            else:
                y = dv_layer(cat_y)  # Skip the nonlinear layer
        out = self.final(y)
        # b = self.b(y)
        # s = self.s(y)
        # w = self.w(y)
        if return_p:
            return out, y
        else:
            return out  # , b, s, w

        
# ==============================================================================
# =                         Optimizer and Learning rate                        =
# ==============================================================================    

def lambda_rule(epoch) -> float:
    """ stepwise learning rate calculator """
    exponent = int(np.floor((epoch + 1) / args.decay_step_size))
    return np.power(args.lr_decay_factor, exponent)


def update_lr():
        """ Learning rate updater """
        
        scheduler.step()
        lr = optim.param_groups[0]['lr']
        if lr < args.min_lr:
            optim.param_groups[0]['lr'] = args.min_lr
            lr = optim.param_groups[0]['lr']
        print('Learning rate = %.7f' % lr) 

# ==============================================================================
# =                              Training procedure                            =
# ==============================================================================    

def training_step(batch, y, y_compounds):
    
    optim.zero_grad(set_to_none=True) 
    compounds = batch[:, 0]
    sources = batch[:, 1]

    calibrated_batch = mmd_resnet(batch)

    # Find same compounds in different sources
    # d = torch.cdist(calibrated_batch, calibrated_batch, p=2.)

    # Loop through sources, compare like-to-like compounds from d to y
    loss = None
    unique_compounds = torch.unique(compounds)
    # for idx, co in tqdm(enumerate(unique_compounds), total=len(unique_compounds), desc="Computing loss"):
    
    for idx, co in enumerate(unique_compounds):
        mask = y_compounds == co
        if mask.sum():  # compound is in the label set
            y_data = y[mask]
            l = torch.cdist(y_data, calibrated_batch[compounds == co], p=2.)
            l = l.mean()
            if loss == None:
                loss = l
            else:
                loss = loss + l

    loss.backward()
    optim.step()
    return loss
    
# ==============================================================================
# =                                     Main                                   =
# ==============================================================================    
     
def main(data_path="../datasets/", version=24, num_test=100):

    # accelerator = Accelerator()
    # device = accelerator.device
    npz_dataset = "/media/data/final_data.npz"
    full_dataset = "cce_precached_data.npz"
    if os.path.exists(npz_dataset):  # accelerator.is_local_main_process:  # accelerator.is_main_process:
        data = np.load(npz_dataset, allow_pickle=True)
        train_res, train_source, train_batch, train_well, train_compounds = data["train_res"], data["train_source"], data["train_batch"], data["train_well"], data["train_compounds"]
        test_res, test_source, test_batch, test_well, test_compounds = data["test_res"], data["test_source"], data["test_batch"], data["test_well"], data["test_compounds"]
        orf_data, orf_source, orf_batch, orf_well = data["orf_data"], data["orf_source"], data["orf_batch"], data["orf_well"]
        crispr_data, crispr_source, crispr_batch, crispr_well = data["crispr_data"], data["crispr_source"], data["crispr_batch"], data["crispr_well"]
        res, batch, well, source = data["res"], data["batch"], data["well"], data["source"]
        compounds, smiles, inchis, keys = data["compounds"], data["smiles"], data["inchis"], data["keys"]
        orfs, selfies, crisprs = data["orfs"], data["selfies"], data["crisprs"]

        # Add encodings to the front of the data
        train_res = torch.Tensor(train_res).float()
        train_source = torch.Tensor(train_source).long()
        train_batch = torch.Tensor(train_batch).long()
        # train_plate = torch.Tensor(train_plate).long()
        train_well = torch.Tensor(train_well).long()
        train_compounds = torch.Tensor(train_compounds).long()
        # train_smiles = torch.Tensor(train_smiles)
        test_res = torch.Tensor(test_res).float()
        test_source = torch.Tensor(test_source).long()
        test_batch = torch.Tensor(test_batch).long()
        # test_plate = torch.Tensor(test_plate).long()
        test_well = torch.Tensor(test_well).long()
        test_compounds = torch.Tensor(test_compounds).long()
        orf_data = torch.Tensor(orf_data).float()
        orf_source = torch.Tensor(orf_source).long()
        orf_batch = torch.Tensor(orf_batch).long()
        # orf_plate = torch.Tensor(orf_plate).long()
        orf_well = torch.Tensor(orf_well).long()

        crispr_data = torch.Tensor(crispr_data).float()
        crispr_source = torch.Tensor(crispr_source).long()
        crispr_batch = torch.Tensor(crispr_batch).long()
        crispr_well = torch.Tensor(crispr_well).long()

        res = torch.Tensor(res).float()
        batch = torch.Tensor(batch).long()
        source = torch.Tensor(source).long()
        well = torch.Tensor(well).long()

        # sample1_tensor = torch.Tensor(train_X).float()# .to(device)
        # test_sample1_tensor = torch.Tensor(test_X).float()# .to(device)
        train_dataset = torch.utils.data.TensorDataset(train_res, train_compounds, train_source, train_batch, train_well)
        test_dataset = torch.utils.data.TensorDataset(test_res, test_compounds, test_source, test_batch, test_well)
        if len(orf_source) != len(orf_data):  # Not clear why this would happen but it has happened
            orf_source = orf_source[:len(orf_data)]
        orf_idx = torch.arange(len(orfs))
        crispr_idx = torch.arange(len(crisprs))
        orf_dataset = torch.utils.data.TensorDataset(orf_data, orf_source, orf_batch, orf_well, orf_idx)
        crispr_dataset = torch.utils.data.TensorDataset(crispr_data, crispr_source, crispr_batch, crispr_well, crispr_idx)

        class_sample_count = np.array(
            [len(np.where(train_compounds == t)[0]) for t in np.unique(train_compounds)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in train_compounds])
        samples_weight = torch.from_numpy(samples_weight)

        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler)
            # num_workers=8)
            # shuffle=True)
        out_train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=100,
            shuffle=False)
        orf_loader = torch.utils.data.DataLoader(
            orf_dataset,
            batch_size=args.batch_size,
            shuffle=False)
        orf_loader = torch.utils.data.DataLoader(
            orf_dataset,
            batch_size=args.batch_size,
            shuffle=False)
        crispr_loader = torch.utils.data.DataLoader(
            crispr_dataset,
            batch_size=args.batch_size,
            shuffle=False)

        emb_dim_b = 64
        emb_dim_s = 16  # 4
        emb_dim_p = 16
        emb_dim_w = 16
        num_embeddings_b = train_batch.max() + 1  # train_batch.shape[-1]
        num_embeddings_s = train_source.max() + 1  # train_source.shape[-1]
        num_embeddings_w = train_well.max() + 1  # train_well.shape[-1]
        # num_embeddings_p = train_plate.shape[-1]
        # comp_emb_dim = train_encodings.shape[-1]
        input_dim = train_res.shape[-1]
        output_dim = train_compounds.max() + 1
        epochs = 2000
        best_loss = 10000000
        epoch_counter = 0
        balanced_loss = False
        args.n_blocks = 6  # 5
        int_dim = 1024  # 600 # input_dim + emb_dim
        tops = 10  # int((train_compounds.max() + 1) * 0.0001)  # Hit-threshold for successfully recovering a compound
        nc = len(np.unique(train_compounds))
        path = "invariant_model.pt"

        model = Mmd_resnet(input_dim, int_dim, output_dim,
                        args.n_blocks, num_embeddings_b=num_embeddings_b, num_embeddings_s=num_embeddings_s, num_embeddings_w=num_embeddings_w, embedding_dim_b=emb_dim_b, embedding_dim_s=emb_dim_s, embedding_dim_w=emb_dim_w)

        optimizer = torch.optim.AdamW(model.parameters(),
                        weight_decay=1e-6,
                        lr=args.lr)  # ,
                        # weight_decay=args.weight_decay)
        scheduler = get_cosine_schedule_with_warmup(  # get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=500, num_training_steps=epochs * int(len(train_loader) // args.batch_size)
        )
    else:
        raise RuntimeError("File not found yet.")

    accelerator = Accelerator()
    device = accelerator.device

    # Run a baseline xgboost
    model, optimizer, train_loader, test_loader, orf_loader, crispr_loader, scheduler, res, source, batch, well = accelerator.prepare(model, optimizer, train_loader, test_loader, orf_loader, crispr_loader, scheduler, res, source, batch, well)
    model.to(device)
    res = res.to(device)
    source = source.to(device)
    batch = batch.to(device)
    well = well.to(device)
    avg_loss = torch.tensor(0).float().to(device)
    best_test_acc = 0.

    accelerator.wait_for_everyone()
    # Get ORF predictions
    # inv_cdict = {v: k for k, v in cdict.items()}
    prep = {k.replace("module.", ""): v for k, v in torch.load(path).items()}
    model.load_state_dict(prep)  # Load best weights
    model.eval()
    with torch.no_grad():
        comp_data = []
        validate, acc = [], []
        for batch_idx, ibatch in enumerate(test_loader):  # tqdm(enumerate(sample1_loader), total=len(sample1_loader), desc="Epoch"):
            dv, text_embeddings, iv_s, iv_b, iv_w = ibatch
            image_embeddings, penultimate = model(dv=dv, iv_s=iv_s, iv_b=iv_b, iv_w=iv_w, return_p=True)
            # image_embeddings = model(dv=dv, iv_s=iv_s, iv_b=iv_b, iv_w=iv_w)
            comp_data.append(penultimate.cpu().numpy())
            loss = nn.CrossEntropyLoss()(image_embeddings, text_embeddings)
            _, tk = torch.topk(image_embeddings, tops, dim=1)
            accuracy = (tk == text_embeddings[:, None]).sum(1).float().sum() / len(tk)
            validate.append(loss.cpu())
            acc.append(accuracy.cpu())
        comp_data = np.concatenate(comp_data, 0)
        print("Final test loss {} and acc {}".format(np.mean(validate), np.mean(acc) * 100.))

        train_comp_data, train_compounds = [], []
        train_keys, train_source, train_well = [], [], []
        num_batches = np.ceil(len(res) / args.batch_size).astype(int)
        batches = np.arange(num_batches).reshape(-1, 1).repeat(args.batch_size, axis=0).ravel()
        batches = batches[:len(res)]
        for idx in range(num_batches):
            batch_idx = batches == idx
            dv, text_embeddings, iv_s, iv_b, iv_w = res[batch_idx], compounds[batch_idx], source[batch_idx], batch[batch_idx], well[batch_idx]
            ks = keys[batch_idx]
            image_embeddings, penultimate = model(dv=dv, iv_s=iv_s, iv_b=iv_b, iv_w=iv_w, return_p=True)
            train_comp_data.append(penultimate.cpu().numpy())
            train_compounds.append(text_embeddings)
            train_keys.append(ks)
            train_source.append(iv_s.cpu().numpy())
            train_well.append(iv_w.cpu().numpy())
        train_comp_data = np.concatenate(train_comp_data, 0)
        train_compounds = np.concatenate(train_compounds)
        train_keys = np.concatenate(train_keys)
        train_source = np.concatenate(train_source)
        train_well = np.concatenate(train_well)

        comp_orf_data, rec_orfs = [], []
        for batch_idx, ibatch in enumerate(orf_loader):  # tqdm(enumerate(sample1_loader), total=len(sample1_loader), desc="Epoch"):
            dv, iv_s, iv_b, iv_w, idx = ibatch
            i_orfs = orfs[idx.cpu().numpy()]
            image_embeddings, penultimate = model(dv=dv, iv_s=iv_s, iv_b=iv_b, iv_w=iv_w, return_p=True)
            comp_orf_data.append(penultimate.cpu().numpy())
            rec_orfs.append(i_orfs)
        comp_orf_data = np.concatenate(comp_orf_data, 0)
        assert (orfs == np.concatenate(rec_orfs)).mean() == 1, "Original vs. reconstructed ORFs are different."
        comp_crispr_data, rec_crisprs = [], []
        for batch_idx, ibatch in enumerate(crispr_loader):  # tqdm(enumerate(sample1_loader), total=len(sample1_loader), desc="Epoch"):
            dv, iv_s, iv_b, iv_w, idx = ibatch
            i_crisprs = crisprs[idx.cpu().numpy()]
            image_embeddings, penultimate = model(dv=dv, iv_s=iv_s, iv_b=iv_b, iv_w=iv_w, return_p=True)
            comp_crispr_data.append(penultimate.cpu().numpy())
            rec_crisprs.append(i_crisprs)
        comp_crispr_data = np.concatenate(comp_crispr_data, 0)
        assert (crisprs == np.concatenate(rec_crisprs)).mean() == 1, "Original vs. reconstructed ORFs are different."

    np.savez("batch_data-{}".format(version), comp_data=train_comp_data, compounds=train_compounds, test_comp_data=comp_data, orfs=orfs, orf_data=comp_orf_data, test_compounds=test_compounds, norm_comp_data=res.cpu().numpy(), norm_orf_data=orf_data, orf_source=orf_source, orf_batch=orf_batch, orf_well=orf_well, norm_crispr_data=crispr_data, crispr_source=crispr_source, crispr_batch=crispr_batch, crispr_well=crispr_well, crisprs=crisprs, crispr_data=comp_crispr_data, keys=train_keys, source=train_source, well=train_well, smiles=smiles, selfies=selfies, inchis=inchis)


if __name__ == '__main__':
    main()

