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
from scipy.stats import iqr
from scipy.spatial.distance import cdist, squareform
from accelerate import Accelerator
import selfies as sf
from joblib import Parallel, delayed, Memory
from rdkit import Chem

import random
import string


# ==============================================================================
# =                                Input arguments                            =
# ==============================================================================

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--n_blocks', 
                    type=int, 
                    default=5, 
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


def aggregate_data(agg_fun, res, source, batch, plate, well, compounds):
    uc = np.unique(compounds)
    ares, asource, aencodings, abatch, aplate, awell, acompounds = [], [], [], [], [], [], []
    for u in uc:
        idx = u == compounds
        ares.append(agg_fun(res[idx]))
        asource.append(source[idx][0])
        # aencodings.append(agg_fun(encodings[idx]))
        abatch.append(batch[idx][0])
        aplate.append(plate[idx][0])
        awell.append(well[idx][0])
        acompounds.append(compounds[idx][0])
    res = np.concatenate(ares, 0)
    source = np.asarray(asource)
    # encodings = np.concatenate(aencodings)
    batch = np.asarray(abatch)
    well = np.asarray(awell)
    compounds = np.asarray(acompounds)
    return res, source, batch, plate, well, compounds


def unique_map(x):
    # u, inv = np.unique(x, return_inverse=True)
    u = np.unique(x)
    h = {i: idx for idx, i in enumerate(u)}
    inv = [h[t] for t in x]
    return inv, h


if not os.path.exists(os.path.join(os.getcwd(), "joblib_cache")):
    os.makedirs(os.path.join(os.getcwd(), "joblib_cache"))
memory = Memory(os.path.join(os.getcwd(), "joblib_cache"), verbose=0)
@memory.cache
def pubchem_smiles(inch, smile_conv):
    check = smile_conv['0'] == inch
    if check.sum():
        smile = smile_conv[check]['1'].values[0]
    else:
        smile = "nan"
    return smile


# @memory.cache
def inchi_to_smiles(inchi):
    try:
        smiles = Chem.MolToSmiles(Chem.inchi.MolFromInchi(inchi), isomericSmiles=False, kekuleSmiles=True, canonical=True)
    except:
        smiles = "nan"
    return smiles


from pubchempy import get_compounds

# @memory.cache
def convert_inchi_smiles(x):
    try:
        cs = get_compounds(x, "inchi")[0]
        smiles = cs.canonical_smiles
    except:
        smiles = "nan"
    return smiles


@memory.cache
def sf_encoder(x):
     try:
         s = sf.encoder(x)
     except:
         s = "nan"
     return s


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
                # torch.nn.Linear(int_dim + embedding_dim_b + embedding_dim_s, int_dim),
                torch.nn.Linear(int_dim + embedding_dim_s, int_dim),
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
        b = self.b(y)
        s = self.s(y)
        w = self.w(y)
        if return_p:
            return out, y
        else:
            return out, b, s, w

        
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
    

def nIQR(x):
    med = np.median(x, 0)
    viqr = iqr(x, axis=0)
    return (x - med) / viqr


def df_norm(df, col):
    group, data = df
    data = data.values[:, col:]
    denom = iqr(data, axis=0) + 1e-2
    ndata = (data - np.median(data, 0)) / denom  # Norm function
    return ndata


def df_meta(df, col):
    group, data = df
    data = data.values[:, :col]
    return data


# ==============================================================================
# =                                     Main                                   =
# ==============================================================================    
     
def main(data_path="../datasets/", version=24, num_test=100):

    accelerator = Accelerator()
    device = accelerator.device
    npz_dataset = "cce_precached_data.npz"
    final_data = "/media/data/final_data.npz"
    force_data = False
    split = "orf_source"
    eval_orf_distance = False
    zscore = False
    if 1:  # accelerator.is_local_main_process:  # accelerator.is_main_process:
        if not os.path.exists(npz_dataset):
            # Load blocklist
            # blocklist = pd.read_table("blocklist.txt")

            # Load orf genes
            orf_filename = os.path.join(data_path, "orf_data-{}.csv".format(version))  # "orf_data.csv"
            orf_col = 18
            print("Reading ORF data")
            orf_df = pd.read_csv(orf_filename, engine="pyarrow")

            # Group by median well cellprofiler
            orf_df = orf_df.groupby(["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well", "Metadata_Symbol"])[orf_df.columns[orf_col:]].median().reset_index()

            # Normalize
            # orf_df = orf_df[orf_df.columns[orf_col:]] = orf_df.groupby(["Metadata_Source", "Metadata_Batch", "Metadata_Plate"])[orf_df.columns[orf_col:]].transform(nIQR)
            print("Transforming ORF data")
            orf_col = 5
            groups = orf_df.groupby(["Metadata_Source", "Metadata_Batch", "Metadata_Plate"])
            orf_data = Parallel(n_jobs=-1)(delayed(df_norm)(x, orf_col) for x in tqdm(groups, total=len(groups), desc="Normalizing ORF data"))
            orf_meta = Parallel(n_jobs=-1)(delayed(df_meta)(x, orf_col) for x in tqdm(groups, total=len(groups), desc="Gethering ORF metadata"))
            orf_data = [np.concatenate((m, d), 1) for m, d in zip(orf_meta, orf_data)]
            orf_data = np.concatenate(orf_data, 0)
            orf_cols = [x for x in groups][0][1].columns
            orf_df = pd.DataFrame(orf_data, columns=orf_cols)

            # med, iquart = groups[orf_df.columns[orf_col:]].transform("median"), groups[orf_df.columns[orf_col:]].transform(lambda x: iqr(x, axis=0))
            # orf_df[orf_df.columns[orf_col:]] = (orf_df[orf_df.columns[orf_col:]] - med) / iquart

            # Process
            orf_data = orf_df.values[:, orf_col:].astype(np.float32)  # 1-20 are metadata, remainder are cellprofiler
            # orf_cols = orf_df.columns[orf_col:]
            orfs = orf_df.Metadata_Symbol.values.astype(str)

            # Load orf genes
            crispr_filename = os.path.join(data_path, "crispr_data-{}.csv".format(version))  # "orf_data.csv"
            crispr_col = 9
            print("Reading CRISPRi data")
            crispr_df = pd.read_csv(crispr_filename, engine="pyarrow")

            # Group by median well cellprofiler
            crispr_df = crispr_df.groupby(["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well", "Metadata_Symbol"])[crispr_df.columns[crispr_col:]].median().reset_index()

            # Normalize
            print("Transforming CRISPRi data")
            crispr_col = 5
            groups = crispr_df.groupby(["Metadata_Source", "Metadata_Batch", "Metadata_Plate"])
            crispr_data = Parallel(n_jobs=-1)(delayed(df_norm)(x, crispr_col) for x in tqdm(groups, total=len(groups), desc="Normalizing CRISPRi data"))
            crispr_meta = Parallel(n_jobs=-1)(delayed(df_meta)(x, crispr_col) for x in tqdm(groups, total=len(groups), desc="Gethering CRISPRi metadata"))
            crispr_data = [np.concatenate((m, d), 1) for m, d in zip(crispr_meta, crispr_data)]
            crispr_data = np.concatenate(crispr_data, 0)
            crispr_cols = [x for x in groups][0][1].columns
            crispr_df = pd.DataFrame(crispr_data, columns=crispr_cols)

            # Process
            crispr_data = crispr_df.values[:, crispr_col:].astype(np.float32)  # 1-20 are metadata, remainder are cellprofiler
            crisprs = crispr_df.Metadata_Symbol.values.astype(str)

            print("Reading Comp data")
            # X = np.load("normalized_batch_data.npy")
            # Load compounds
            compound_filename = os.path.join(data_path, "compound_data-{}.csv".format(version))
            comp_col = 9
            comp_df = pd.read_csv(compound_filename, engine="pyarrow")

            # Group by median well cellprofiler
            comp_df = comp_df.groupby(["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well", "Metadata_InChIKey", "Metadata_InChI"])[comp_df.columns[comp_col:]].median().reset_index()
            comp_col = 6

            # Normalize
            # comp_df[comp_df.columns[comp_col:]] = comp_df.groupby(["Metadata_Source", "Metadata_Batch", "Metadata_Plate"])[comp_df.columns[comp_col:]].transform(nIQR)
            print("Transforming Compound data")
            groups = comp_df.groupby(["Metadata_Source", "Metadata_Batch", "Metadata_Plate"])
            comp_cols = [x for x in groups][0][1].columns
            comp_data = Parallel(n_jobs=-1)(delayed(df_norm)(x, comp_col) for x in tqdm(groups, total=len(groups), desc="Normalizing COMP data"))
            comp_meta = Parallel(n_jobs=-1)(delayed(df_meta)(x, comp_col) for x in tqdm(groups, total=len(groups), desc="Gethering COMP metadata"))
            comp_data = [np.concatenate((m, d), 1) for m, d in zip(comp_meta, comp_data)]
            comp_data = np.concatenate(comp_data, 0)
            comp_cols = [x for x in groups][0][1].columns
            comp_df = pd.DataFrame(comp_data, columns=comp_cols)

            # med, iquart = groups[comp_df.columns[comp_col:]].transform("median"), groups[comp_df.columns[comp_col:]].transform(lambda x: iqr(x, axis=0))
            # comp_df[comp_df.columns[comp_col:]] = (comp_df[comp_df.columns[comp_col:]] - med) / iquart

            # Process
            print("Filtering compounds")
            comp_df = comp_df[~comp_df.Metadata_InChI.isnull()]
            comp_df = comp_df[np.logical_or(~comp_df.Metadata_InChI.isnull(), comp_df.Metadata_InChI != "")]
            res = comp_df.values[:, comp_col:].astype(np.float32)

            # Mask outlier features
            # mask = np.logical_or(res > 100, res < -100).sum(0) == 0
            res = np.clip(res, -30, 30)  # Let's clip instead
            orf_data = np.clip(orf_data, -30, 30)
            crispr_data = np.clip(crispr_data, -30, 30)

            # Find and remove fluerescent compounds by using quinine as a reference
            quinine_idx = comp_df.Metadata_InChI == "InChI=1S/C20H24N2O2/c1-3-13-12-22-9-7-14(13)10-19(22)20(23)16-6-8-21-18-5-4-15(24-2)11-17(16)18/h3-6,8,11,13-14,19-20,23H,1,7,9-10,12H2,2H3"
            res = res[~quinine_idx]  # Just kill Quinine
            comp_df = comp_df[~quinine_idx]
            """
            quinine_features = res[quinine_idx]
            other_features = res[~quinine_idx]
            other_inchis = comp_df.Metadata_InChI[~quinine_idx].values
            # [x for x in comp_df.columns[7:] if "Intensity_MeanIntensity_" in x]
            col_mask = []
            for c in comp_df.columns[7:]:
                if "Intensity_MeanIntensity_" in c:
                    col_mask.append(True)
                else:
                    col_mask.append(False)
            col_mask = np.asarray(col_mask)
            dm = cdist(quinine_features[:, col_mask], other_features[:, col_mask])
            qarg = np.argsort(dm)
            qinch = other_inchis[qarg]
            """

            # jcp = comp_df.Metadata_JCP2022.values
            source = comp_df.Metadata_Source.values
            plate = comp_df.Metadata_Plate.values
            well = comp_df.Metadata_Well.values
            batch = comp_df.Metadata_Batch.values
            keys = comp_df.Metadata_InChIKey.values  # Pos/neg controls: https://github.com/jump-cellpainting/JUMP-Target
            inchis = comp_df.Metadata_InChI.values

            compounds, comp_hash = unique_map(inchis)
            # jcp, jcp_hash = unique_map(jcp)
            source, source_hash = unique_map(source)
            plate, plate_hash = unique_map(plate)
            batch, batch_hash = unique_map(batch)
            well, well_hash = unique_map(well)

            # keys, keys_hash = unique_map(keys)
            compounds = np.asarray(compounds)
            source = np.asarray(source)
            plate = np.asarray(plate)
            batch = np.asarray(batch)
            well = np.asarray(well)
            keys = np.asarray(keys)

            # Convert to SMILES
            # smile_conv = pd.read_csv("../CLIP_prefix_caption/pubchem_smiles.csv")
            if 0:  # os.path.exists("smiles_cache.npy"):
                smiles = np.load("smiles_cache.npy")
            else:
                # cache, smiles = {}, []
                # for inch in tqdm(inchi, total=len(inchi)):
                #     if inch in cache:
                #         smiles.append(cache[inch])
                #     else:
                #         check = smile_conv['0'] == inch
                #         if check.sum():
                #             smile = smile_conv[check]['1'].values[0]
                #         else:
                #             smile = "nan"
                #         smiles.append(smile)
                #         cache[inch] = smile
                # smiles = Parallel(n_jobs=-1)(delayed(pubchem_smiles)(x, smile_conv) for x in tqdm(inchi, total=len(inchi), desc="Converting inchis to pubchem"))
                # smiles = Parallel(n_jobs=-1)(delayed(convert_inchi_smiles)(x) for x in tqdm(inchi, total=len(inchi), desc="Converting inchis to pubchem"))
                smiles = []
                cache = {}
                for i in tqdm(inchis, total=len(inchis), desc="Converting inchi to smiles"):
                    if i in cache:
                        smiles.append(cache[i])
                    else:
                        smile = inchi_to_smiles(i)
                        # try:
                        #     smile = convert_inchi_smiles(i)
                        # except:
                        #     smile = "nan"
                        cache[i] = smile
                        smiles.append(smile)
                smiles = np.asarray(smiles)
                np.save("smiles_cache.npy", smiles)
            non_empty_compounds = smiles != "nan"
            inchis = inchis[non_empty_compounds]
            smiles = smiles[non_empty_compounds]
            res = res[non_empty_compounds]
            compounds = compounds[non_empty_compounds]
            source = source[non_empty_compounds]
            plate = plate[non_empty_compounds]
            batch = batch[non_empty_compounds]
            well = well[non_empty_compounds]
            keys = keys[non_empty_compounds]

            if 0:  # os.path.exists("selfies_cache.npy"):
                selfies = np.load("selfies_cache.npy")
            else:
                selfies, mask = [], []
                for s in tqdm(smiles, total=len(smiles)):
                    try:
                        selfies.append(sf.encoder(s))
                        mask.append(True)
                    except:
                        selfies.append("nan")
                        mask.append(False)
                # selfies = Parallel(n_jobs=-1)(delayed(sf_encoder)(x) for x in tqdm(smiles, total=len(smiles), desc="Converting smiles to selfies"))
                selfies = np.asarray(selfies)
                mask = selfies != "nan"
                mask = np.asarray(mask)
                np.save("selfies_cache.npy", selfies)
            mask = selfies != "nan"
            mask = np.asarray(mask)
            inchis = inchis[mask] #non_empty_compounds]
            selfies = selfies[mask]  # non_empty_compounds]
            smiles = smiles[mask]
            res = res[mask]  # non_empty_compounds]
            compounds = compounds[mask]  # non_empty_compounds]
            source = source[mask]  # non_empty_compounds]
            plate = plate[mask]  # non_empty_compounds]
            batch = batch[mask]  # non_empty_compounds]
            well = well[mask]  # non_empty_compounds]
            keys = keys[mask]

            # print("Loading compound encodings")
            # encodings = np.load("ncfrey_cache.npy").squeeze()
            # encodings = np.load("ncfrey_cache_max.npy").squeeze()

            # Save everything
            orf_source = np.asarray([source_hash[x] for x in orf_df.Metadata_Source.values])
            orf_well = np.asarray([well_hash[x] for x in orf_df.Metadata_Well.values])
            orf_batch = np.asarray([batch_hash[x] for x in orf_df.Metadata_Batch.values])
            crispr_source = np.asarray([source_hash[x] for x in crispr_df.Metadata_Source.values])
            crispr_well = np.asarray([well_hash[x] for x in crispr_df.Metadata_Well.values])
            crispr_batch = np.asarray([batch_hash[x] for x in crispr_df.Metadata_Batch.values])
            # mask, orf_plate = [], []
            # for p in orf_df.Metadata_Plate.values:
            #     if p in batch_hash:
            #         orf_plate.append(batch_hash[p])
            #         mask.append(True)
            #     else:
            #         mask.append(False)
            # orf_plate, mask = np.asarray(orf_plate), np.asarray(mask)
            # print("Keeping {} ORFs".format(mask.sum()))
            # orf_data = orf_data[mask]
            # ords = orfs[mask]
            # orf_source = orf_source[mask]
            # orf_well = orf_well[mask]
            # orf_batch = orf_batch[mask]

            # orf_plate = np.asarray([batch_hash[x] for x in orf_df.Metadata_Plate.values])
            # orf_X = np.stack((source, well, batch), 1).astype(np.float32)
            # orf_X = source   # np.stack((source, well, batch), 1).astype(np.float32)
            # orf_X = np.concatenate((orf_X, orf_data), 1)

            # Z-score each batch  DEPRECIATED -- NORMALIZING EARLIER
            outliers = 4  # Sigma cutoff for outliers
            t_thresh = 2.1
            if zscore:  # https://www.biorxiv.org/content/10.1101/2023.05.01.538999v1.full.pdf
                if not os.path.exists("zdata.npz"):
                    uni_comp = np.unique(source)
                    means, stds = {}, {}
                    for so in tqdm(uni_comp, desc="Zscoring", total=len(uni_comp)):
                        idx = so == source
                        data = res[idx]
                        # # mu = data.mean(0)
                        # # sd = data.std(0)
                        # mu = np.median(data, 0)
                        # sd = iqr(data, axis=0)
                        # z = (data - mu) / sd
                        # flt = (z > outliers).sum(1) != 0  # Find where outliers are for Nans

                        # # Also do a distance matrix outlier check
                        # dm = np.abs(np.corrcoef(data))
                        # tt = dm.mean(1) / dm.std(1)
                        # tt = tt < t_thresh  # p > 0.05 to remove later
                        # data[flt] = np.nan  # Remove these later later

                        # mu = np.nanmean(data, 0)
                        # sd = np.nanstd(data, 0)
                        mu = np.nanmedian(data, 0)
                        sd = iqr(data, axis=0)
                        means[so] = mu
                        stds[so] = sd
                        data = (data - mu) / sd
                        res[idx] = data
                    np.savez("zdata.npz", res=res, means=means, stds=stds)
                else:
                    d = np.load("zdata.npz", allow_pickle=True)
                    res = d["res"]
                    means = d["means"].item()
                    stds = d["stds"].item()

            source = np.asarray(source)
            compounds = np.asarray(compounds)

            np.savez(
                npz_dataset,
                res=res,
                source=source,
                # encodings=encodings,
                batch=batch,
                plate=plate,
                well=well,
                keys=keys,
                compounds=compounds,
                selfies=selfies,
                smiles=smiles,
                inchis=inchis,  # comp_df,
                orf_data=orf_data,
                orf_source=orf_source,
                orf_batch=orf_batch,
                crispr_data=crispr_data,
                crispr_source=crispr_source,
                crispr_batch=crispr_batch,
                # orf_plate=orf_plate,
                comp_hash=comp_hash,
                source_hash=source_hash,
                orf_well=orf_well,
                orfs=orfs,
                crispr_well=crispr_well,
                crisprs=crisprs)
            print("Finished preprocessing data.")
            finished_final_data = False
        else:
            if not os.path.exists(final_data) or force_data:  # accelerator.is_main_process:
                data = np.load(npz_dataset, allow_pickle=True)
                comp_hash = data["comp_hash"].item()
                source_hash = data["source_hash"].item()
                res, source, batch, plate, well, compounds, orf_data, orf_source, orf_batch, orf_well, orfs = data["res"], data["source"], data["batch"], data["plate"], data["well"], data["compounds"], data["orf_data"], data["orf_source"], data["orf_batch"], data["orf_well"], data["orfs"]
                selfies, smiles, inchis, keys = data["selfies"], data["smiles"], data["inchis"], data["keys"]
                crispr_data, crispr_source, crispr_batch, crispr_well, crisprs = data["crispr_data"], data["crispr_source"], data["crispr_batch"], data["crispr_well"], data["crisprs"]
                if len(smiles) != len(selfies):
                    raise RuntimeError
                    # Reconstruct smiles
                    smiles = Parallel(n_jobs=-1)(delayed(sf.decoder)(x) for x in tqdm(selfies, total=len(selfies), desc="Reconstructing smiles"))
                    smiles = np.asarray(smiles)
                    # smiles,selfies
                """
                # Reconstruct Inchis and Keys for later. This needs to be fixed in the first preprocessing step eventually.
                print("Reading Comp data")
                # Load compounds
                compound_filename = os.path.join(data_path, "compound_data-{}.csv".format(version))
                comp_col = 9
                comp_df = pd.read_csv(compound_filename, engine="pyarrow")

                # Group by median well cellprofiler
                comp_df = comp_df.groupby(["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well", "Metadata_InChIKey", "Metadata_InChI"]).median().reset_index()
                comp_col = 7

                print("Filtering compounds")
                comp_df = comp_df[~comp_df.Metadata_InChI.isnull()]
                comp_df = comp_df[np.logical_or(~comp_df.Metadata_InChI.isnull(), comp_df.Metadata_InChI != "")]
                quinine_idx = comp_df.Metadata_InChI == "InChI=1S/C20H24N2O2/c1-3-13-12-22-9-7-14(13)10-19(22)20(23)16-6-8-21-18-5-4-15(24-2)11-17(16)18/h3-6,8,11,13-14,19-20,23H,1,7,9-10,12H2,2H3"
                comp_df = comp_df[~quinine_idx]
                # an_keys = comp_df.Metadata_InChIKey.values  # Pos/neg controls: https://github.com/jump-cellpainting/JUMP-Target
                inchis = comp_df.Metadata_InChI.values
                """
                finished_final_data = False
            else:
                finished_final_data = True

        if not finished_final_data:
            # Mask nan columns
            mask = np.logical_and(np.logical_and(np.isnan(res).sum(0) == 0, np.isnan(orf_data).sum(0) == 0), np.isnan(crispr_data).sum(0) == 0)
            res = res[:, mask]
            orf_data = orf_data[:, mask]
            crispr_data = crispr_data[:, mask]

            # Mask bad rows from compounds
            flt = np.isnan(res).sum(1) == 0
            res = res[flt]  # Keep non-outliers
            source = source[flt]
            well = well[flt]
            plate = plate[flt]
            batch = batch[flt]
            compounds = compounds[flt]
            keys = keys[flt]
            selfies = selfies[flt]
            smiles = smiles[flt]
            inchis = inchis[flt]
            # an_keys = an_keys[flt]

            # print("Running PCA enc")
            from sklearn.decomposition import PCA
            print("Running PCA res")
            res_model = PCA(n_components=.999, whiten=True)
            res = res_model.fit_transform(res)

            if zscore:
                orf_data = orf_data[flt]
                orf_source = orf_source[flt]
                orf_batch = orf_batch[flt]
                orfs = orfs[flt]
                orf_data = (orf_data - means[orf_source[0]]) / stds[orf_source[0]]
            orf_data = res_model.transform(orf_data)
            crispr_data = res_model.transform(crispr_data)

            # Filter train set for compounds that are over-represented (these are shitty controls)
            print("Filtering over-represented compounds")
            co_thresh = 10000
            if co_thresh:
                uc, co = np.unique(compounds, return_counts=True)
                uc = uc[co < co_thresh]
                kcomp, kres, ksource, kencodings, kbatch, kwell = [], [], [], [], [], []
                kselfies, ksmiles, kkeys = [], [], []
                kinchi = []
                for u in uc:
                    idx = u == compounds
                    kcomp.append(compounds[idx])
                    kres.append(res[idx])
                    ksource.append(source[idx])
                    kbatch.append(batch[idx])
                    kwell.append(well[idx])
                    kselfies.append(selfies[idx])
                    ksmiles.append(smiles[idx])
                    kkeys.append(keys[idx])
                    kinchi.append(inchis[idx])
                    # kankeys.append(an_keys[idx])
                compounds = np.concatenate(kcomp)
                res = np.concatenate(kres, 0)
                source = np.concatenate(ksource)
                batch = np.concatenate(kbatch)
                well = np.concatenate(kwell)
                selfies = np.concatenate(kselfies)
                smiles = np.concatenate(ksmiles)
                keys = np.concatenate(kkeys)
                inchis = np.concatenate(kinchi)
                # an_keys = np.concatenate(kankeys)

            # Filter sources
            filter_sources = True
            sel_src = orf_source[0]  # Use the same source as orf
            if filter_sources:
                print("Filtering sources")
                mask = source == sel_src
                us, sc = np.unique(source, return_counts=True)
                idx = np.argsort(sc)[::-1]
                us = us[idx]
                top_counts = 9  # 5  # Introduce this many of the sources with the most compounds
                for u in us[:top_counts]:
                    mask = np.logical_or(mask, source == u)
                # mask = np.logical_or(mask, source == 11)
                # mask = np.logical_or(mask, source == 10)
                compounds = compounds[mask]
                res = res[mask]
                source = source[mask]
                # encodings = encodings[mask]
                batch = batch[mask]
                well = well[mask]
                selfies = selfies[mask]
                smiles = smiles[mask]
                keys = keys[mask]
                inchis = inchis[mask]
                # an_keys = an_keys[mask]

            # Renumber compounds from 0
            compounds, cdict = renumber(compounds, return_dict=True)
            batch = renumber(batch)
            well = renumber(well)
            source, sdict = renumber(source, return_dict=True)

            # Split into train/test
            print("Splitting train from test")
            # agg_fun = lambda x: np.mean(x, 0)
            up_src = sdict[orf_source[0]]  # Take just the sel_src/orf_source[0] compounds for the test dict
            # uc = np.unique(compounds)
            if split == "orf_source":
                uc, counts = np.unique(compounds[source == up_src], return_counts=True)
            else:
                uc, counts = np.unique(compounds, return_counts=True)
            test_X_idx, train_X_idx = [], []
            # num_comps = 500000000  # 500
            # uc = uc[:num_comps]
            # uc = uc[np.logical_or(counts < 100, counts > 20)]
            for u in uc:
                idx = u == compounds
                if split == "orf_source":
                    comb = np.logical_and(idx, source == up_src)
                else:
                    comb = idx
                if len(comb):
                    uw, wc = np.unique(well[comb], return_counts=True)
                    if len(uw) > 1:
                        sel_wll = uw[1]
                    else:
                        sel_wll = uw
                    test_ids = np.where(np.logical_and(comb, well == sel_wll))[0]  # Use images from only this well
                    test_X_idx.append(test_ids)
            test_X_idx = np.concatenate(test_X_idx)
            train_X_idx = np.arange(len(compounds))[~np.in1d(np.arange(len(compounds)), test_X_idx)]

            # Save testing metadata
            metatest = "test_meta.npz"
            # rk = random_key(5)
            # metatest = "{}_{}".format(rk, metatest)
            # comp_names = [inv_comp_hash[cdict[x]] for x in compounds[test_X_idx]]
            # source_names = [inv_source_hash[sdict[x]] for x in source[test_X_idx]]
            np.savez(
                metatest,
                test_X_idx=test_X_idx,
                test_source=source[test_X_idx],
                test_compounds=compounds[test_X_idx],
                test_selfies=selfies[test_X_idx],
                test_smiles=smiles[test_X_idx],
                test_well=well[test_X_idx])  # , comp_names=comp_names, source_names=source_names)

            # Double check sources
            test_sources = source[test_X_idx]
            if split == "orf_source":
                assert len(np.unique(test_sources)) == 1, "Multiple sources found in test data."
                assert test_sources[0] == up_src, "Test data includes sources besides the ORF-generator."
            assert not np.any(np.in1d(train_X_idx, test_X_idx)), "Overlap between training and testing data."

            # And split sets
            train_res = res[train_X_idx]
            train_source = source[train_X_idx]
            train_batch = batch[train_X_idx]
            train_plate = plate[train_X_idx]
            train_well = well[train_X_idx]
            train_compounds = compounds[train_X_idx]

            test_res = res[test_X_idx]
            test_source = source[test_X_idx]
            test_batch = batch[test_X_idx]
            test_plate = plate[test_X_idx]
            test_well = well[test_X_idx]
            test_compounds = compounds[test_X_idx]

            # And optionally aggregate
            # agg_fun = lambda x: np.median(x, 0, keepdims=True)
            agg_fun = False
            if agg_fun:
                # train_res, train_source, train_encodings, train_batch, train_plate, train_well, train_compounds = aggregate_data(agg_fun, train_res, train_source, train_encodings, train_batch, train_plate, train_well, train_compounds)
                test_res, test_source, test_batch, test_plate, test_well, test_compounds = aggregate_data(agg_fun, test_res, test_source, test_batch, test_plate, test_well, test_compounds)
            np.savez(
                final_data,  # "final_data.npz",
                train_res=train_res,
                train_source=train_source,
                train_batch=train_batch,
                train_well=train_well,
                train_compounds=train_compounds,
                test_res=test_res,
                test_source=test_source,
                test_batch=test_batch,
                test_well=test_well,
                test_compounds=test_compounds,
                orf_source=orf_source,
                orf_batch=orf_batch,
                orf_well=orf_well,
                orfs=orfs,
                crispr_source=crispr_source,
                crispr_batch=crispr_batch,
                crispr_well=crispr_well,
                crisprs=crisprs,
                res=res,
                batch=batch,
                plate=plate,
                well=well,
                source=source,
                smiles=smiles,
                selfies=selfies,
                keys=keys,
                inchis=inchis,
                test_X_idx=test_X_idx,
                compounds=compounds,
                orf_data=orf_data,
                crispr_data=crispr_data)
            os._exit(1)
        else:
            data = np.load(final_data)
            train_res, train_source, train_batch, train_well, train_compounds = data["train_res"], data["train_source"], data["train_batch"], data["train_well"], data["train_compounds"]
            test_res, test_source, test_batch, test_well, test_compounds = data["test_res"], data["test_source"], data["test_batch"], data["test_well"], data["test_compounds"]
            orf_data, orf_source, orf_batch, orf_well = data["orf_data"], data["orf_source"], data["orf_batch"], data["orf_well"]
            crispr_data, crispr_source, crispr_batch, crispr_well = data["orf_data"], data["orf_source"], data["orf_batch"], data["orf_well"]
            res, orfs, crisprs = data["res"], data["orfs"], data["crisprs"]

        # Inverse weighting
        uni_c, class_sample_count = np.unique(train_compounds, return_counts=True)
        weight = 1. / class_sample_count
        weight_dict = {k: v for k, v in zip(uni_c, weight)}
        samples_weight = np.array([weight_dict[t] for t in train_compounds])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        # Add encodings to the front of the data
        train_res = torch.Tensor(train_res).float()
        train_source = torch.Tensor(train_source).long()
        # train_encodings = torch.Tensor(train_encodings).float()
        train_batch = torch.Tensor(train_batch).long()
        # train_plate = torch.Tensor(train_plate).long()
        train_well = torch.Tensor(train_well).long()
        train_compounds = torch.Tensor(train_compounds).long()
        test_res = torch.Tensor(test_res).float()
        test_source = torch.Tensor(test_source).long()
        # test_encodings = torch.Tensor(test_encodings).float()
        test_batch = torch.Tensor(test_batch).long()
        # test_plate = torch.Tensor(test_plate).long()
        test_well = torch.Tensor(test_well).long()
        test_compounds = torch.Tensor(test_compounds).long()
        orf_data = torch.Tensor(orf_data).float()
        orf_source = torch.Tensor(orf_source).long()
        orf_batch = torch.Tensor(orf_batch).long()
        unique_orfs, orf_idx = np.unique(orfs, return_inverse=True)
        orf_idx = torch.Tensor(orf_idx).long()
        # orf_plate = torch.Tensor(orf_plate).long()
        orf_well = torch.Tensor(orf_well).long()
        crispr_data = torch.Tensor(crispr_data).float()
        crispr_source = torch.Tensor(crispr_source).long()
        crispr_batch = torch.Tensor(crispr_batch).long()
        unique_crisprs, crispr_idx = np.unique(crisprs, return_inverse=True)
        crispr_idx = torch.Tensor(crispr_idx).long()
        crispr_well = torch.Tensor(crispr_well).long()

        # sample1_tensor = torch.Tensor(train_X).float()# .to(device)
        # test_sample1_tensor = torch.Tensor(test_X).float()# .to(device)
        train_dataset = torch.utils.data.TensorDataset(train_res, train_compounds, train_source, train_batch, train_well)
        test_dataset = torch.utils.data.TensorDataset(test_res, test_compounds, test_source, test_batch, test_well)
        # print(len(test_res))
        if len(orf_source) != len(orf_data):  # Not clear why this would happen but it has happened
            orf_source = orf_source[:len(orf_data)]
        orf_dataset = torch.utils.data.TensorDataset(orf_data, orf_idx, orf_source, orf_batch, orf_well)
        crispr_dataset = torch.utils.data.TensorDataset(crispr_data, crispr_idx, crispr_source, crispr_batch, crispr_well)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            drop_last=True,
            batch_size=args.batch_size,
            sampler=sampler)
            # num_workers=8)
            # shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=34,
            drop_last=True,
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

        # accelerator.wait_for_everyone()

        model = Mmd_resnet(input_dim, int_dim, output_dim,
                        args.n_blocks, num_embeddings_b=num_embeddings_b, num_embeddings_s=num_embeddings_s, num_embeddings_w=num_embeddings_w, embedding_dim_b=emb_dim_b, embedding_dim_s=emb_dim_s, embedding_dim_w=emb_dim_w)
        # model = MLP(input_dim, int_dim, output_dim,
        #                      args.n_blocks, num_embeddings_b=num_embeddings_b, num_embeddings_s=num_embeddings_s, num_embeddings_w=num_embeddings_w, embedding_dim_b=emb_dim_b, embedding_dim_s=emb_dim_s, embedding_dim_w=emb_dim_w)

        optimizer = torch.optim.AdamW(model.parameters(),
                        weight_decay=1e-6,
                        lr=args.lr)  # ,
                        # weight_decay=args.weight_decay)
        scheduler = get_cosine_schedule_with_warmup(  # get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=500, num_training_steps=epochs * int(len(train_loader) // args.batch_size)
        )

    device = accelerator.device

    # Run a baseline xgboost
    model, optimizer, train_loader, test_loader, orf_loader, crispr_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, test_loader, orf_loader, crispr_loader, scheduler)
    model.to(device)
    avg_loss = torch.tensor(0).float().to(device)
    best_test_acc = 0.

    accelerator.wait_for_everyone()

    eb = None
    teb = None
    for epoch in range(epochs):  # count(1):
        batch_losses = []
        progress = tqdm(total=len(train_loader), desc="Training", disable=not accelerator.is_local_main_process)
        model.train()
        for batch_idx, batch in enumerate(train_loader):  # tqdm(enumerate(sample1_loader), total=len(sample1_loader), desc="Epoch"):
            # batch = batch.to(device=device)
            # batch2 = batch2[0].to(device=device)
            
            optimizer.zero_grad(set_to_none=True)
            # batch = batch[0]
            dv, text_embeddings, iv_s, iv_b, iv_w = batch
            image_embeddings, b, s, w = model(dv=dv, iv_s=iv_s, iv_b=iv_b, iv_w=iv_w)
            # image_embeddings = model(dv=dv, iv_s=iv_s, iv_b=iv_b, iv_w=iv_w)

            # Make entropic targets
            if eb is None:
                eb = F.softmax(torch.ones_like(b), 1)
                es = F.softmax(torch.ones_like(s), 1)
                ew = F.softmax(torch.ones_like(w), 1)
            loss = nn.CrossEntropyLoss()(image_embeddings, text_embeddings)
            bl = F.cross_entropy(b, eb)
            sl = F.cross_entropy(s, es)
            wl = F.cross_entropy(w, ew)
            loss = loss + bl + sl + wl

            # Optimize
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            batch_losses.append(loss)
            progress.set_postfix({"train_loss": loss})  # , "compounds": comp_loss, "phenotypes": pheno_loss})
            progress.update()

        # Run test set
        test_losses, test_accs = [], []
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):  # tqdm(enumerate(sample1_loader), total=len(sample1_loader), desc="Epoch"):
                dv, text_embeddings, iv_s, iv_b, iv_w = batch
                image_embeddings, b, s, w = model(dv=dv, iv_s=iv_s, iv_b=iv_b, iv_w=iv_w)
                # image_embeddings = model(dv=dv, iv_s=iv_s, iv_b=iv_b, iv_w=iv_w)
                loss = nn.CrossEntropyLoss()(image_embeddings, text_embeddings)
                # Losses to become invariant to batch effects
                if teb is None:
                    teb = F.softmax(torch.ones_like(b), 1)
                    tes = F.softmax(torch.ones_like(s) / len(s), 1)
                    tew = F.softmax(torch.ones_like(w) / len(w), 1)

                bl = (b.argmax(1) == iv_b).float().mean()  # F.cross_entropy(b, teb)
                sl = (s.argmax(1) == iv_s).float().mean()  # F.cross_entropy(s, tes)
                wl = (w.argmax(1) == iv_w).float().mean()  # F.cross_entropy(w, tew)
                _, tk = torch.topk(image_embeddings, tops, dim=1)
                accuracy = (tk == text_embeddings[:, None]).sum(1).float().sum() / len(tk)
                test_losses.append(loss)
                test_accs.append(accuracy)

        # Check ORF distances
        if eval_orf_distance:
            orf_losses = []
            model.eval()
            with torch.no_grad():
                for batch_idx, batch in enumerate(orf_loader):  # tqdm(enumerate(sample1_loader), total=len(sample1_loader), desc="Epoch"):
                    dv, oidx, iv_s, iv_b, iv_w = batch
                    image_embeddings, b, s, w = model(dv=dv, iv_s=iv_s, iv_b=iv_b, iv_w=iv_w)
                    # image_embeddings = model(dv=dv, iv_s=iv_s, iv_b=iv_b, iv_w=iv_w)
                    loss = 0
                    uo = oidx.unique()
                    dm = torch.cdist(image_embeddings, image_embeddings)
                    for u in uo:
                        uidx = oidx == u
                        loss = loss + dm[uidx, uidx].mean()
                    loss = loss / len(uo)
                    orf_losses.append(loss)

        # Check performances
        epoch_loss = np.mean([x.item() for x in batch_losses])
        test_loss = np.mean([x.item() for x in test_losses])
        test_acc = np.mean([x.item() for x in test_accs]) * 100.
        if eval_orf_distance:
            orf_losses = np.mean([x.item() for x in orf_losses])
        if accelerator.is_main_process:
            if test_loss < best_loss:
                print("Saving best performing weights")
                best_loss = test_loss
                best_test_acc = test_acc
                torch.save(model.state_dict(), path)
                epoch_counter = 0
            else:
                epoch_counter += 1
            if eval_orf_distance:
                progress.set_postfix({"epoch": epoch, "number_compounds": nc, "train_loss": epoch_loss, "test_loss": test_loss, "test_acc": test_acc, "best_test_acc": best_test_acc, "orf_dist": orf_losses, "well_acc": wl, "batch_acc": bl, "source_acc": sl})
            else:
                progress.set_postfix({"epoch": epoch, "number_compounds": nc, "train_loss": epoch_loss, "test_loss": test_loss, "test_acc": test_acc, "best_test_acc": best_test_acc, "well_loss": wl, "batch_loss": bl, "source_loss": sl})
            progress.update()
        # if epoch_counter == args.epochs_wo_im:
        #     break
        progress.close()
        accelerator.wait_for_everyone()

    print('Finished training')
    
    
if __name__ == '__main__':
    main()

