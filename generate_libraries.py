import os
import shutil
import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import iqr
from glob import glob


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


# Hyperparams
fn = "batch_data-18.npz"
d = np.load(fn, allow_pickle=True)
metric = "euclidean"
targets = [
    # "TRKB",
    "TARDBP",
    "PKD2",
    "APP",
    "SNCA",
    "DRD1",
    "DRD5",
]
cache_fn = "cached_target_compound_preds.npz"
norm = "iqr"  # "zscore"  # "iqr"  # 
agg = "min"
version = "0.3.1"
top_k = 10000000  # Analyze these for targets/moas
rerun = True

# Load and combine repurposing spreadsheets
df = pd.read_csv("repurpose_database_X_jump.csv")
rephub = pd.read_csv("compound_rephub_annot_full.csv.gz")
df_target = flatten(df, "target")
df_moa = flatten(df, "moa")
df_inchi_key = flatten(df, "inchi_key")
rephub_target = flatten(rephub, "target")
rephub_moa = flatten(rephub, "moa")
rephub_inchi_key = flatten(rephub, "InChIKey")

df_target_mask = df_target != "nan"
df_moa_mask = df_moa != "nan"
df_inchi_key_mask = df_inchi_key != "nan"
rephub_target_mask = rephub_target != "nan"
rephub_moa_mask = rephub_moa != "nan"
rephub_inchi_key_mask = rephub_inchi_key != "nan"
df_mask = df_target_mask & df_moa_mask & df_inchi_key_mask
rephub_mask = rephub_target_mask & rephub_moa_mask & rephub_inchi_key_mask

df_target = df_target[df_mask]
df_moa = df_moa[df_mask]
df_inchi_key = df_inchi_key[df_mask]
rephub_target = rephub_target[rephub_mask]
rephub_moa = rephub_moa[rephub_mask]
rephub_inchi_key = rephub_inchi_key[rephub_mask]

df1 = np.stack((df_target, df_moa, df_inchi_key), 1)
df2 = np.stack((rephub_target, rephub_moa, rephub_inchi_key), 1)
df = pd.DataFrame(np.concatenate((df1, df2), 0), columns=["target", "moa", "inchi_key"])
# df = df.groupby("inchi_key").transform(lambda x: ','.join(x)).reset_index()
mask_targ = np.asarray([len(x) for x in df.target.values]) > 0  # Remove empties 
mask_moa = np.asarray([len(x) for x in df.moa.values]) > 0  # Remove empties 
mask = np.logical_and(mask_targ, mask_moa)
df = df[mask]

# Prep data
if not os.path.exists(cache_fn):
    # Load data
    well = d["well"]
    source = d["source"]
    data = d["comp_data"]
    control_data = d["norm_comp_data"]
    comps = d["compounds"]
    inchis = d["inchis"]
    smiles = d["smiles"]
    keys = d["keys"]
    orfs = d["orfs"]
    control_orf_data = d["norm_orf_data"]
    orf_data = d["orf_data"]
    orf_well = d["orf_well"]
    ck_map = np.load("comp_key_map.npy", allow_pickle=True).item()
    inv_ck_map = {v: k for k, v in ck_map.items()}

    # Compound performances
    ucomp = np.unique(inchis)
    res, cres, cont_res, ccomps, csmiles, ckeys, cinchis = [], [], [], [], [], [], []
    for co in tqdm(ucomp, total=len(ucomp), desc="Aggregating Compounds"):
        idx = inchis == co
        # ikey = ck_map[co]
        ismiles = smiles[idx]
        ikeys = keys[idx]
        iinchis = inchis[idx]
        wl = well[idx]
        uw = np.unique(wl)
        for w in uw:
            # widx = well == uw
            cidx = np.logical_and(idx, well == w)
            ccomps.append(co)
            csmiles.append(ismiles[0])
            ckeys.append(ikeys[0])
            cinchis.append(iinchis[0])
            res.append(data[cidx].mean(0))
            cont_res.append(control_data[cidx].mean(0))
            cres.append(co)
    res = np.stack(res, 0)
    cont_res = np.stack(cont_res, 0)
    cres = np.asarray(cres)
    ccomps = np.asarray(ccomps)
    csmiles = np.asarray(csmiles)
    ckeys = np.asarray(ckeys)
    cinchis = np.asarray(cinchis)

    # ORF performances
    uorfs = np.unique(orfs)
    orf_res, corf, orf_cont_res = [], [], []
    for co in tqdm(uorfs, total=len(uorfs), desc="Aggregating ORFs"):
        idx = orfs == co
        wl = orf_well[idx]
        uw = np.unique(wl)
        for w in uw:
            # widx = well == uw
            cidx = np.logical_and(idx, orf_well == w)
            orf_res.append(orf_data[cidx].mean(0))
            orf_cont_res.append(control_orf_data[cidx].mean(0))
            corf.append(co)
    orf_res = np.stack(orf_res, 0)
    orf_cont_res = np.stack(orf_cont_res, 0)
    corf = np.asarray(corf)
    np.savez(
        cache_fn,
        res=res,
        cont_res=cont_res,
        cres=cres,
        ckeys=ckeys,
        csmiles=csmiles,
        cinchis=cinchis,
        ccomps=ccomps,
        orf_res=orf_res,
        orf_cont_res=orf_cont_res,
        corf=corf)
else:
    d = np.load(cache_fn)
    res, cont_res, cres, orf_res, orf_cont_res, corf = d["res"], d["cont_res"], d["cres"], d["orf_res"], d["orf_cont_res"], d["corf"]
    ccomps, ckeys, csmiles, cinchis = d["ccomps"], d["ckeys"], d["csmiles"], d["cinchis"]

##### DEBUG — USING UNPOOLED ORF DATA
d = np.load(fn, allow_pickle=True)
control_orf_data = d["norm_orf_data"]
orf_data = d["orf_data"]
orfs = d["orfs"]
orf_res, orf_cont_res, corf = orf_data, control_orf_data, orfs

# Delete empties
mask = corf != ""
orf_res = orf_res[mask]
orf_cont_res = orf_cont_res[mask]
corf = corf[mask]

if norm == "zscore":
    # Normalize compounds
    res = (res - res.mean(0)) / res.std(0)
    cont_res = (cont_res - cont_res.mean(0)) / cont_res.std(0)

    # Normalize orfs
    orf_res = (orf_res - orf_res.mean(0)) / orf_res.std(0)
    orf_cont_res = (orf_cont_res - orf_cont_res.mean(0)) / orf_cont_res.std(0)
elif norm == "iqr":
    # Normalize compounds
    res = (res - np.median(res, 0)) / iqr(res, 0)
    cont_res = (cont_res - np.median(cont_res, 0)) / iqr(cont_res, 0)

    # Normalize orfs
    orf_res = (orf_res - np.median(orf_res, 0)) / iqr(orf_res, 0)
    orf_cont_res = (orf_cont_res - np.median(orf_cont_res, 0)) / iqr(orf_cont_res, 0)
else:
    raise NotImplementedError

# Get repurposed compound libraries for each target
lib_compounds, lib_smiles, lib_inchis, lib_keys, lib_targets, lib_moas, lib_scores = {}, {}, {}, {}, {}, {}, {}
for t in tqdm(targets, total=len(targets), desc="Annotating target libraries"):
    idx = t == corf

    # Our version
    Xa = orf_res[idx]
    if agg == "mean":
        dm = cdist(Xa, res, metric=metric).mean(0)  # Take average dist
    elif agg == "median":
        dm = np.median(cdist(Xa, res, metric=metric), 0)  # Take min dist
    elif agg == "min":
        dm = cdist(Xa, res, metric=metric).min(0)  # Take min dist
    else:
        raise NotImplementedError(agg)
    srt = np.argsort(dm)
    srt_dm = dm[srt]
    comps = cres[srt]  # Compound library in inchi
    icomps = ccomps[srt]  # Compound library in keys
    ikeys = ckeys[srt]  # Compound library in keys
    ismiles = csmiles[srt]  # Compound library in smiles
    iinchis = cinchis[srt]  # Compound library in smiles

    # # COMMENTED FOR JOHN -- REVERT THIS SOON
    # trgs, moas = [], []
    # for k in tqdm(ikeys[:top_k], total=len(ikeys), desc="Annotating our model"):
    #     trg = df[df.inchi_key == k].target.values
    #     moa = df[df.inchi_key == k].moa.values
    #     if not len(trg):
    #         trg = ""
    #     else:
    #         trg = np.unique(trg)
    #     if not len(moa):
    #         moa = ""
    #     else:
    #         moa = np.unique(moa)
    #     trgs.append(trg)
    #     moas.append(moa)

    # Control version
    Xa = orf_cont_res[idx]
    if agg == "mean":
        dm = cdist(Xa, cont_res, metric=metric).mean(0)  # Take average dist
    elif agg == "min":
        dm = cdist(Xa, cont_res, metric=metric).min(0)  # Take min dist
    else:
        raise NotImplementedError(agg)
    srt = np.argsort(dm)
    cont_comps = cres[srt]  # Compound library in inchi
    cont_icomps = ccomps[srt]  # Compound library in keys
    cont_ikeys = ckeys[srt]  # Compound library in keys
    cont_ismiles = csmiles[srt]  # Compound library in smiles
    cont_iinchis = cinchis[srt]  # Compound library in smiles
    cont_srt_dm = dm[srt]

    # cont_trgs, cont_moas = [], []
    # for k in tqdm(ikeys[:top_k], total=len(ikeys), desc="Annotating control model"):
    #     trg = df[df.inchi_key == k].target.values
    #     moa = df[df.inchi_key == k].moa.values
    #     if not len(trg):
    #         trg = ""
    #     else:
    #         trg = np.unique(trg)
    #     if not len(moa):
    #         moa = ""
    #     else:
    #         moa = np.unique(moa)
    #     cont_trgs.append(trg)
    #     cont_moas.append(moa)

    # trgs, moas = np.asarray(trgs), np.asarray(moas)
    # cont_trgs, cont_moas = np.asarray(cont_trgs), np.asarray(cont_moas)
    lib_compounds[t] = comps
    lib_smiles[t] = ismiles
    lib_inchis[t] = iinchis
    lib_keys[t] = ikeys
    # lib_targets[t] = trgs
    # lib_moas[t] = moas
    lib_scores[t] = srt_dm

    t = "control_{}".format(t)
    lib_compounds[t] = cont_comps
    lib_smiles[t] = cont_ismiles
    lib_inchis[t] = cont_iinchis
    lib_keys[t] = cont_ikeys
    # lib_targets[t] = cont_trgs
    # lib_moas[t] = cont_moas
    lib_scores[t] = cont_srt_dm

# Screens
lib_df = pd.DataFrame.from_dict(lib_compounds)
lib_df.to_csv("{}_compound_library_v{}.csv".format(norm, version))
smiles_df = pd.DataFrame.from_dict(lib_smiles)
smiles_df.to_csv("{}_compound_smiles_v{}.csv".format(norm, version))
inchis_df = pd.DataFrame.from_dict(lib_inchis)
inchis_df.to_csv("{}_compound_inchis_v{}.csv".format(norm, version))
keys_df = pd.DataFrame.from_dict(lib_keys)
keys_df.to_csv("{}_compound_keys_v{}.csv".format(norm, version))
# targ_df = pd.DataFrame.from_dict(lib_targets)
# targ_df.to_csv("{}_compound_targets_v{}.csv".format(norm, version))
# moa_df = pd.DataFrame.from_dict(lib_moas)
# moa_df.to_csv("{}_compound_moas_v{}.csv".format(norm, version))
score_df = pd.DataFrame.from_dict(lib_scores)
score_df.to_csv("{}_compound_scores_v{}.csv".format(norm, version))

# !mv *_compound_* /media/data_cifs/clicktionary/clickme_experiment/tf_records/
files = glob("{}_compound_*_v{}.csv".format(norm, version))
[shutil.move(f, "/media/data_cifs/clicktionary/clickme_experiment/tf_records/") for f in files]

