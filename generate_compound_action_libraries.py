import os
import shutil
import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import iqr, spearmanr
from glob import glob
from joblib import Parallel, delayed


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


def manip_math(manip, wt_vector, res, return_scores=False, metric="cosine", direction="other"):
    if direction == "plus":
        Xa = manip + res  # Compound (Res) should move manipulation towards wt (DMSO). Seq2Seq logic: Sick+healthy = healthy
        di = cdist(Xa, wt_vector, metric=metric)  # Distance between modulated orf + wild type
        gdi = di.min(1)
    elif direction == "minus":
        Xa = manip + wt_vector  # This should be as close as possible to res
        di = cdist(Xa, res, metric=metric)
        gdi = di.min(0)  # Greedy search for best match-per modulated orf
    elif direction == "other":
        # Xa = wt_vector - manip  # This should be as close as possible to res
        Xa = manip - wt_vector  # Compound (Res) should move manipulation towards wt (DMSO). Seq2Seq logic: Sick+healthy = healthy
        di = cdist(Xa, res, metric=metric)
        gdi = di.min(0)  # Greedy search for best match-per modulated orf
    else:
        raise NotImplementedError(direction)
    if return_scores:
        return np.argsort(gdi), gdi
    else:
        return np.argsort(gdi)


# Hyperparams
fn = "batch_data-24.npz"
d = np.load(fn, allow_pickle=True)
gen_manipulation = "CRISPR"
metric = "euclidean"
targets = [
    # "PKD2",
    # "SRC",
    # "RAF1",
    # "MYB",
    # "FOS",
    "SOD1",
    "SOD2"
]
cache_fn = "cached_compound_action.npz"
norm = "iqr"  # "zscore"  # "iqr"  # 
agg = "min"
version = "0.4.1"
top_k = 10000000  # Analyze these for targets/moas
rerun = True

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

    crisprs = d["crisprs"]
    control_crispr_data = d["norm_crispr_data"]
    crispr_data = d["crispr_data"]
    crispr_well = d["crispr_well"]

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

    # CRISPR performances
    ucrisprs = np.unique(crisprs)
    crispr_res, ccrispr, crispr_cont_res = [], [], []
    for co in tqdm(ucrisprs, total=len(ucrisprs), desc="Aggregating CRISPRs"):
        idx = crisprs == co
        wl = crispr_well[idx]
        uw = np.unique(wl)
        for w in uw:
            # widx = well == uw
            cidx = np.logical_and(idx, crispr_well == w)
            crispr_res.append(crispr_data[cidx].mean(0))
            crispr_cont_res.append(control_crispr_data[cidx].mean(0))
            ccrispr.append(co)
    crispr_res = np.stack(crispr_res, 0)
    crispr_cont_res = np.stack(crispr_cont_res, 0)
    ccrispr = np.asarray(ccrispr)
    np.savez(
        cache_fn,
        res=res,
        cont_res=cont_res,
        cres=cres,
        ckeys=ckeys,
        csmiles=csmiles,
        cinchis=cinchis,
        ccomps=ccomps,
        crispr_res=crispr_res,
        crispr_cont_res=crispr_cont_res,
        ccrispr=ccrispr,
        orf_res=orf_res,
        orf_cont_res=orf_cont_res,
        corf=corf)
else:
    d = np.load(cache_fn)
    res, cont_res, cres, orf_res, orf_cont_res, corf = d["res"], d["cont_res"], d["cres"], d["orf_res"], d["orf_cont_res"], d["corf"]
    crispr_res, crispr_cont_res, ccrispr = d["crispr_res"], d["crispr_cont_res"], d["ccrispr"]
    ccomps, ckeys, csmiles, cinchis = d["ccomps"], d["ckeys"], d["csmiles"], d["cinchis"]

##### DEBUG — USING UNPOOLED GENETIC DATA
d = np.load(fn, allow_pickle=True)
control_orf_data = d["norm_orf_data"]
orf_data = d["orf_data"]
orfs = d["orfs"]
orf_res, orf_cont_res, corf = orf_data, control_orf_data, orfs

control_crispr_data = d["norm_crispr_data"]
crispr_data = d["crispr_data"]
crisprs = d["crisprs"]
crispr_res, crispr_cont_res, ccrispr = crispr_data, control_crispr_data, crisprs
# plt.imshow(np.corrcoef(np.concatenate((crispr_data[crisprs == "SOD1"], crispr_data[crisprs == "SOD2"], crispr_data[crisprs == "CP"]), 0)));plt.show()

# Delete empties
mask = corf != ""
orf_res = orf_res[mask]
orf_cont_res = orf_cont_res[mask]
corf = corf[mask]

mask = ccrispr != ""
crispr_res = crispr_res[mask]
crispr_cont_res = crispr_cont_res[mask]
ccrispr = ccrispr[mask]

if norm == "zscore":
    # Normalize compounds
    res = (res - res.mean(0)) / res.std(0)
    cont_res = (cont_res - cont_res.mean(0)) / cont_res.std(0)

    # Normalize orfs
    orf_res = (orf_res - orf_res.mean(0)) / orf_res.std(0)
    orf_cont_res = (orf_cont_res - orf_cont_res.mean(0)) / orf_cont_res.std(0)

    # Normalize crisprs
    crispr_res = (crispr_res - crispr_res.mean(0)) / crispr_res.std(0)
    crispr_cont_res = (crispr_cont_res - crispr_cont_res.mean(0)) / crispr_cont_res.std(0)
elif norm == "iqr":
    # Normalize compounds
    res = (res - np.median(res, 0)) / iqr(res, 0)
    cont_res = (cont_res - np.median(cont_res, 0)) / iqr(cont_res, 0)

    # Normalize orfs
    orf_res = (orf_res - np.median(orf_res, 0)) / iqr(orf_res, 0)
    orf_cont_res = (orf_cont_res - np.median(orf_cont_res, 0)) / iqr(orf_cont_res, 0)

    # Normalize crisprs
    crispr_res = (crispr_res - np.median(crispr_res, 0)) / iqr(crispr_res, 0)
    crispr_cont_res = (crispr_cont_res - np.median(crispr_cont_res, 0)) / iqr(crispr_cont_res, 0)
else:
    raise NotImplementedError

# Get repurposed compound libraries for each target
if gen_manipulation.lower() == "orf":
    gen_names = corf
    gen_res = orf_res
    gen_cont_res = orf_cont_res
    wt_vector = gen_res[gen_names == "LacZ"]
    cont_wt_vector = gen_cont_res[gen_names == "LacZ"]
elif gen_manipulation.lower() == "crispr":
    gen_names = ccrispr
    gen_res = crispr_res
    gen_cont_res = crispr_cont_res
    wt_vector = np.median(gen_res[gen_names == "non-targeting"], 0, keepdims=True)
    cont_wt_vector = np.median(gen_cont_res[gen_names == "non-targeting"], 0, keepdims=True)
else:
   raise NotImplementedError(gen_manipulation)

"""
idx = "SOD1" == gen_names
manip = gen_res[idx]

metric="cosine"
Xa = manip + wt_vector  # This should be as close as possible to res        
di = cdist(Xa, res, metric=metric)       
sod1_gdi = di.min(0)  # Greedy search for best match-per modulated orf

idx = "SOD2" == gen_names
manip = gen_res[idx]

Xa = manip + wt_vector  # This should be as close as possible to res        
di = cdist(Xa, res, metric=metric)
sod2_gdi = di.min(0)  # Greedy search for best match-per modulated orf


idx = gen_names[100] == gen_names
manip = gen_res[idx]

Xa = manip + wt_vector  # This should be as close as possible to res        
di = cdist(Xa, res, metric=metric)
pkd_gdi = di.min(0)  # Greedy search for best match-per modulated orf


import pdb;pdb.set_trace()
corr = spearmanr(sod1_gdi, sod2_gdi)
null_corr_a = spearmanr(sod1_gdi, pkd_gdi)
null_corr_b = spearmanr(sod2_gdi, pkd_gdi)
"""
# corr = cdist(gen_res["SOD1" == gen_names], gen_res["SOD2" == gen_names], "euclidean")
# null_corr = cdist(gen_res["SOD1" == gen_names], gen_res["TP53" == gen_names], "euclidean")

lib_compounds, lib_smiles, lib_inchis, lib_keys, lib_targets, lib_moas, lib_scores = {}, {}, {}, {}, {}, {}, {}
for t in tqdm(targets, total=len(targets), desc="Annotating target libraries"):
    idx = t == gen_names

    # Our version
    igen_manipulation = gen_res[idx]

    # Look through each manipulation and argmin the distance
    dms = Parallel(n_jobs=-1)(delayed(manip_math)(gen_manip, wt_vector, res) for gen_manip in igen_manipulation)
    # dms, scs = manip_math(np.median(igen_manipulation, 0), wt_vector, res, True)
    comb = np.stack(dms, 1)  # Combine the sorts into a matrix
    hsh = {}  # Get the median ordering of each compound
    for i in range(comb.shape[1]):
        for j in range(comb.shape[0]):
            k = comb[j, i]
            if k not in hsh:
                hsh[k] = []
            hsh[k].append(j)
    ranks = {k: int(np.median(v)) for k, v in hsh.items()}
    rank_order = np.argsort([k for k in ranks.values()])
    srt = np.asarray([k for k in ranks.keys()])[rank_order]

    # Use the median rank for each compound
    # srt_dm = dm[srt]
    comps = cres[srt]  # Compound library in inchi
    icomps = ccomps[srt]  # Compound library in keys
    ikeys = ckeys[srt]  # Compound library in keys
    ismiles = csmiles[srt]  # Compound library in smiles
    iinchis = cinchis[srt]  # Compound library in smiles

    # Control version
    igen_manipulation = gen_cont_res[idx]

    # Look through each manipulation and argmin the distance
    dms = Parallel(n_jobs=-1)(delayed(manip_math)(gen_manip, cont_wt_vector, cont_res) for gen_manip in igen_manipulation)
    comb = np.stack(dms, 1)  # Combine the sorts into a matrix
    hsh = {}  # Get the median ordering of each compound
    for i in range(comb.shape[1]):
        for j in range(comb.shape[0]):
            k = comb[j, i]
            if k not in hsh:
                hsh[k] = []
            hsh[k].append(j)
    ranks = {k: int(np.median(v)) for k, v in hsh.items()}
    rank_order = np.argsort([k for k in ranks.values()])
    srt = np.asarray([k for k in ranks.keys()])[rank_order]
    cont_comps = cres[srt]  # Compound library in inchi
    cont_icomps = ccomps[srt]  # Compound library in keys
    cont_ikeys = ckeys[srt]  # Compound library in keys
    cont_ismiles = csmiles[srt]  # Compound library in smiles
    cont_iinchis = cinchis[srt]  # Compound library in smiles
    # cont_srt_dm = dm[srt]

    lib_compounds[t] = comps
    lib_smiles[t] = ismiles
    lib_inchis[t] = iinchis
    lib_keys[t] = ikeys

    t = "control_{}".format(t)
    lib_compounds[t] = cont_comps
    lib_smiles[t] = cont_ismiles
    lib_inchis[t] = cont_iinchis
    lib_keys[t] = cont_ikeys

# Screens
lib_df = pd.DataFrame.from_dict(lib_compounds)
lib_df.to_csv("{}_compound_action_library_v{}.csv".format(gen_manipulation, version))
smiles_df = pd.DataFrame.from_dict(lib_smiles)
smiles_df.to_csv("{}_compound_action_smiles_v{}.csv".format(gen_manipulation, version))
inchis_df = pd.DataFrame.from_dict(lib_inchis)
inchis_df.to_csv("{}_compound_action_inchis_v{}.csv".format(gen_manipulation, version))
keys_df = pd.DataFrame.from_dict(lib_keys)
keys_df.to_csv("{}_compound_action_keys_v{}.csv".format(gen_manipulation, version))
score_df = pd.DataFrame.from_dict(lib_scores)
score_df.to_csv("{}_compound_action_scores_v{}.csv".format(gen_manipulation, version))

# !mv *_compound_* /media/data_cifs/clicktionary/clickme_experiment/tf_records/
files = glob("{}_compound_action_*_v{}.csv".format(gen_manipulation, version))
[shutil.move(f, "/media/data_cifs/clicktionary/clickme_experiment/tf_records/") for f in files]

