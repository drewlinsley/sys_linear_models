"""UNTESTED

Rely on data generated from the original repo.
"""

import os
import random
import string
import argparse
from joblib import Parallel, delayed, Memory

import numpy as np
import pandas as pd

from sklearn import decomposition
from itertools import count
from functools import partial

from scipy.stats import iqr
from scipy.spatial.distance import cdist, squareform

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.utils.data.sampler import WeightedRandomSampler

from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from accelerate import Accelerator

import selfies as sf
from rdkit import Chem

from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)



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
    # np.save("smiles_cache.npy", smiles)
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
    # np.save("selfies_cache.npy", selfies)

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

    # Save everything
    orf_source = np.asarray([source_hash[x] for x in orf_df.Metadata_Source.values])
    orf_well = np.asarray([well_hash[x] for x in orf_df.Metadata_Well.values])
    orf_batch = np.asarray([batch_hash[x] for x in orf_df.Metadata_Batch.values])
    crispr_source = np.asarray([source_hash[x] for x in crispr_df.Metadata_Source.values])
    crispr_well = np.asarray([well_hash[x] for x in crispr_df.Metadata_Well.values])
    crispr_batch = np.asarray([batch_hash[x] for x in crispr_df.Metadata_Batch.values])

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
        compounds = compounds[mask]
        res = res[mask]
        source = source[mask]
        batch = batch[mask]
        well = well[mask]
        selfies = selfies[mask]
        smiles = smiles[mask]
        keys = keys[mask]
        inchis = inchis[mask]

    # Renumber compounds from 0
    compounds, cdict = renumber(compounds, return_dict=True)
    batch = renumber(batch)
    well = renumber(well)
    source, sdict = renumber(source, return_dict=True)

    # Split into train/test
    print("Splitting train from test")
    up_src = sdict[orf_source[0]]  # Take just the sel_src/orf_source[0] compounds for the test dict
    if split == "orf_source":
        uc, counts = np.unique(compounds[source == up_src], return_counts=True)
    else:
        uc, counts = np.unique(compounds, return_counts=True)
    test_X_idx, train_X_idx = [], []
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

