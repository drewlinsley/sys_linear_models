import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind
from scipy.stats import iqr
import seaborn as sns
from matplotlib import patches
from glob import glob
import torch
from sklearn.metrics import balanced_accuracy_score as bas


def flatten(df, col):
    data = df[col].values
    fixed = []
    for d in data:
        if isinstance(d, np.ndarray):
            n = ""
            for r in d:
                n += r
            d = n
        fixed.append(d)
    return np.asarray(fixed)


def main():
    original_data_dir = "../linear_models"
    fn = os.path.join(original_data_dir, "batch_data-24.npz")
    d = np.load(fn, allow_pickle=True)
    well = d["well"]
    source = d["source"]
    data = d["comp_data"]
    control_data = d["norm_comp_data"]
    comps = d["compounds"]
    inchis = d["inchis"]
    smiles = d["smiles"]
    keys = d["keys"]
    metric = "euclidean"
    eval_data_dir = "eval_data"

    # Make an output dir
    os.makedirs(eval_data_dir, exist_ok=True)

    # Load and combine repurposing spreadsheets
    df = pd.read_csv(os.path.join(original_data_dir, "repurpose_database_X_jump.csv"))
    rephub = pd.read_csv(os.path.join(original_data_dir, "compound_rephub_annot_full.csv.gz"))
    antiox = pd.read_csv(os.path.join(original_data_dir, "antioxidants.csv"))

    # Filter antiox
    antiox_inchis = antiox.InChI.values
    filt_antiox_inchi_ids = []
    for x in tqdm(antiox_inchis, total=len(antiox_inchis), desc="Filtering antioxidants"):
        contained = False
        for idx, i in enumerate(inchis):
            if x in i:  # Filter like this to accept partial matches
                contained = True
                break
        if contained:
            filt_antiox_inchi_ids.append(idx)
    filt_antiox_inchi_ids = np.asarray(filt_antiox_inchi_ids)
    filt_antiox_keys = keys[filt_antiox_inchi_ids]

    # Concat filt_antiox_keys to rephub with antioxidant MoA
    moa_lab = np.ones_like(filt_antiox_keys)
    moa_lab[:] = "antioxidant"
    empty_col = np.ones_like(moa_lab)
    empty_col[:] = np.nan
    antiox_info = np.stack((empty_col, empty_col, empty_col, moa_lab, empty_col, empty_col, empty_col, filt_antiox_keys), 1)
    adf = pd.DataFrame(antiox_info, columns=df.columns)
    df = pd.concat((df, adf))

    # Create new df
    df_target = flatten(df, "target")
    df_moa = flatten(df, "moa")
    df_inchi_key = flatten(df, "inchi_key")
    dfs = np.stack((df_target, df_moa, df_inchi_key), 1)

    rephub_target = flatten(rephub, "target")
    rephub_moa = flatten(rephub, "moa")
    rephub_inchi_key = flatten(rephub, "InChIKey")
    rephubs = np.stack((rephub_target, rephub_moa, rephub_inchi_key), 1)

    df_comb = np.concatenate((dfs, rephubs), 0)
    # mask = (df_comb == "nan").sum(1) == 0  # Remove all empty rows
    mask = df_comb[:, 1] != "nan"  # Remove empty MoA rows
    df_comb = df_comb[mask]
    df_comb = pd.DataFrame(df_comb, columns=["target", "moa", "inchi_key"])
    df_comb = df_comb.drop_duplicates()

    # Only keep targets with inchi_keys in keys
    df_key = df_comb.inchi_key
    key_mask = np.in1d(df_key, keys)
    df_comb = df_comb[key_mask]

    # Get moa accuracy for our model and the null model
    moas = df_comb.moa.values.astype(str)  # df.target[~df.target.isnull()].unique()
    umoas = np.unique(moas)
    umoa_len = len(umoas)

    train_y, train_X, cont_train_X = [], [], []
    test_y, test_X, cont_test_X = [], [], []
    train_key, test_key = [], []
    train_inchi, test_inchi = [], []
    train_smile, test_smile = [], []
    train_comps, test_comps = [], []
    compound_moa_remap = {}
    for i, t in tqdm(enumerate(umoas), total=umoa_len, desc="MoAs"):
        ikeys = df_comb[moas == t].inchi_key.values
        idx = np.in1d(keys, ikeys)
        act = data[idx]
        cont_act = control_data[idx]
        ikeys = keys[idx]
        iinchi = inchis[idx]
        iuc = np.unique(comps[idx])
        checks = [x for x in iuc if x not in compound_moa_remap]
        if len(checks):
            for cc in checks:
                compound_moa_remap[cc] = t
        ismile = smiles[idx]
        train_X.append(act[:-1])
        cont_train_X.append(cont_act[:-1])
        test_X.append(act[[-1]])
        cont_test_X.append(cont_act[[-1]])
        train_key.append(ikeys[:-1])
        test_key.append(ikeys[[-1]])
        train_inchi.append(iinchi[:-1])
        test_inchi.append(iinchi[[-1]])
        train_smile.append(ismile[:-1])
        test_smile.append(ismile[[-1]])

        tr_y = [t] * len(ikeys[:-1])
        te_y = [t] * len(ikeys[[-1]])
        train_y.append(tr_y)
        test_y.append(te_y)
    train_X = np.concatenate(train_X)
    cont_train_X = np.concatenate(cont_train_X)
    train_y = np.concatenate(train_y)
    test_X = np.concatenate(test_X)
    cont_test_X = np.concatenate(cont_test_X)
    test_y = np.concatenate(test_y)
    _, train_yi = np.unique(train_y, return_inverse=True)
    _, test_yi = np.unique(test_y, return_inverse=True)
    train_key = np.concatenate(train_key)
    test_key = np.concatenate(test_key)
    train_inchi = np.concatenate(train_inchi)
    test_inchi = np.concatenate(test_inchi)
    train_smile = np.concatenate(train_smile)
    test_smile = np.concatenate(test_smile)

    # Remap Y down to the minimal # of cats
    uni_cats = np.unique(train_y)
    y_names = train_y.copy()
    moa_id_remap = {c: idx for idx, c in enumerate(uni_cats)}

    train_y = np.asarray([moa_id_remap[x] for x in train_y])
    test_y = np.asarray([moa_id_remap[x] for x in test_y])

    # Save metadata
    np.savez(
        os.path.join(eval_data_dir, "moa_data.npz"),
        train_X=train_X,
        train_y=train_y,
        test_X=test_X,
        test_y=test_y,
        train_key=train_key,
        test_key=test_key,
        id_remap=moa_id_remap,
        compound_remap=compound_moa_remap,
        inchi_names=np.unique(train_inchi))
    print("Finished MoA data")

if __name__ == '__main__':
    main()

