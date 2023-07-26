import os
import shutil
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


def ranked_library(
        query,
        db,
        keys,
        smiles,
        metric="euclidean"
    ):
    """Produce a library of phenotypically most similar compounds to the train_q, which has a target or MoA of interest."""
    dm = cdist(query, db, metric=metric)  # Euclidean
    dm_idx = np.argsort(dm, 1)[:, 1:]  # Strip off the first col as it is the same
    comp_idx = keys[dm_idx]
    rank_idx = np.argsort(dm_idx, 1)

    # Now build a consensus ranking
    ranks, scores = {}, {}
    for r in range(dm_idx.shape[0]):
        # Per key compound
        kc = dm_idx[r]
        label = keys[kc]
        scr = dm[r]
        rnk = rank_idx[r]

        for c in range(len(kc)):
            # Per query compound
            comp = label[c]
            rn = rnk[c]
            sc = scr[c]
            if comp in ranks:
                ranks[comp].append(rn)
                scores[comp].append(sc)
            else:
                ranks[comp] = [rn]
                scores[comp] = [sc]

    # Aggregate
    minimum_ranks = {k: int(np.min(v)) for k, v in ranks.items()}
    minimum_scores = {k: np.min(v) for k, v in scores.items()}
    fvals = np.asarray([v for v in minimum_ranks.values()])
    fkeys = np.asarray([k for k in minimum_ranks.keys()])
    fscores = np.asarray([v for v in minimum_scores.values()])
    idx = np.argsort(fvals)
    sorted_keys = fkeys[idx]
    sorted_vals = fvals[idx]
    sorted_scores = fscores[idx]

    # Make a df for sharing
    codebook = {}
    # for k, s in zip(train_key, train_smile):
    for k, s in zip(keys, smiles):
        if k not in codebook:
            codebook[k] = s

    ordered_smiles = []
    for k in sorted_keys:
        ordered_smiles.append(codebook[k])
    ordered_smiles = np.asarray(ordered_smiles)
    out_df = pd.DataFrame(np.stack((sorted_keys, ordered_smiles, sorted_scores, sorted_vals), 1), columns=["inchi_keys", "smiles", "best {} distance".format(metric), "best rank"])
    # Unresolved questions: Is this greedy search over all queries doing more harm than good?
    # What is the right metric? Need some retrieval scores per metric.
    return out_df


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


def knn_acc(null_accs, main_accs, metric, return_random_acc=False):
    """Get KNN acc."""
    mn_ex = min(len(null_accs), len(main_accs))  # Use minimum number of examples between classes
    lab_size = mn_ex - 1
    train_size = lab_size * 2
    X = np.stack((null_accs[:mn_ex], main_accs[:mn_ex]), 1)
    y = np.concatenate((np.zeros(lab_size), np.ones(lab_size))).astype(int)
    rng = np.arange(mn_ex)
    accs, rand_accs = [], []
    for j in range(mn_ex):
        test = X[j]
        train = X[~np.in1d(rng, j)]  # .reshape(train_size, -1)
        dists_a = cdist(test, train[:, 0], metric=metric)
        dists_b = cdist(test, train[:, 1], metric=metric)
        y_hat_a = np.argmin(dists_a, 0)
        y_hat_b = np.argmin(dists_b, 0)
        sums = sum(y_hat_a == 0) + sum(y_hat_b == 1)
        acc = float(sums) / train_size
        if return_random_acc:
            rand_acc = (np.random.rand(train_size) > 0.5).astype(np.float32).mean()
            rand_accs.append(rand_acc)
        accs.append(acc)
    if return_random_acc:
        return np.mean(accs), np.mean(rand_accs)
    return np.mean(accs)


norm = "iqr"
version = "0.0.1"
fn = "batch_data-24.npz"
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

# Train models
model_type = "torch"  # "torch"
if model_type == "torch":
    import torch
    from sklearn.metrics import balanced_accuracy_score as bas

    class TinyModel(torch.nn.Module):

        def __init__(self, out_dims=730, hidden=2048):
            super(TinyModel, self).__init__()
            # self.linear1 = torch.nn.Linear(1024, out_dims)
            self.linear1 = torch.nn.Linear(1024, hidden)
            self.linear2 = torch.nn.Linear(hidden, hidden)
            self.linear3 = torch.nn.Linear(hidden, out_dims)
            self.dropout1 = torch.nn.Dropout(0.1)
            self.dropout2 = torch.nn.Dropout(0.1)
            self.activation = torch.nn.GELU()

        def forward(self, x):
            x = self.linear1(x)
            # x = self.norm(x)
            x = self.dropout1(x)
            x = self.activation(x)
            x = self.linear2(x)
            y = self.dropout2(x)
            y = self.activation(y)
            x = self.linear3(x)
            return x, y


    nc = 942
    model = TinyModel(out_dims=nc).to("cuda")

    # Run training
    best_loss = 100
    moa_dir = "moa_weights"
    os.makedirs(moa_dir, exist_ok=True)
    out_path = os.path.join(moa_dir, "moa_model_{}.pth".format(version))

    # Load best weights
    model.load_state_dict(torch.load(out_path))
    model.eval()

    # data, inchis
    bs = 100000
    nb = np.ceil(len(data) / bs).astype(int)
    batch_idx = np.arange(nb).repeat(bs)[:len(data)]
    with torch.no_grad():
        pX, pY = [], []
        for i in tqdm(np.unique(batch_idx), total=len(np.unique(batch_idx)), desc="Projecting data"):
            idx = batch_idx == i
            if not len(idx):
                continue
            it_X = data[idx]
            it_X = torch.from_numpy(it_X).cuda()
            out, z = model(it_X)
            out = torch.nn.functional.softmax(out, 1)
            pX.append(z.cpu())
            pY.append(out.cpu())
        pX = np.concatenate(pX, 0)
        pY = np.concatenate(pY, 0)

    # Gen libs
    moa_col = 669  # mTOR
    moa_name = "mTOR"
    moa_col = 338
    moa_name = "antioxidant"
    # inv_remap = {v: k for k, v in remap.items()}
    # query_idx = inchis == "InChI=1S/C56H87NO16/c1-33-17-13-12-14-18-34(2)45(68-9)29-41-22-20-39(7)56(67,73-41)51(63)52(64)57-24-16-15-19-42(57)53(65)71-46(30-43(60)35(3)26-38(6)49(62)50(70-11)48(61)37(5)25-33)36(4)27-40-21-23-44(47(28-40)69-10)72-54(66)55(8,31-58)32-59/h12-14,17-18,26,33,36-42,44-47,49-50,58-59,62,67H,15-16,19-25,27-32H2,1-11H3"  # Closest to rapamycin we have
    query_name = "rapamycin"
    query_idx = inchis == "InChI=1S/C18H19N3O3S/c1-21(16-4-2-3-9-19-16)10-11-24-14-7-5-13(6-8-14)12-15-17(22)20-18(23)25-15/h2-9,22H,10-12H2,1H3,(H,20,23)"  # rapalogue
    # query_name = "torin2"
    # query_idx = inchis == "InChI=1S/C24H15F3N4O/c25-24(26,27)17-2-1-3-18(11-17)31-22(32)9-6-16-13-29-20-7-4-14(10-19(20)23(16)31)15-5-8-21(28)30-12-15/h1-13H,(H2,28,30)"
    """
    query_name = "pp242"
    query_idx = inchis == "InChI=1S/C16H16N6O/c1-8(2)22-16-13(15(17)18-7-19-16)14(21-22)12-6-9-5-10(23)3-4-11(9)20-12/h3-8,20,23H,1-2H3,(H2,17,18,19)"
    query_name = "osi027"
    query_idx = inchis == "InChI=1S/C21H22N6O3/c1-30-15-4-2-3-13-9-14(25-16(13)15)17-18-19(22)23-10-24-27(18)20(26-17)11-5-7-12(8-6-11)21(28)29/h2-4,9-12,25H,5-8H2,1H3,(H,28,29)(H2,22,23,24)"
    query_name = "azd8055"
    query_idx = inchis == "InChI=1S/C25H31N5O4/c1-16-14-33-10-8-29(16)24-20-5-6-21(18-4-7-22(32-3)19(12-18)13-31)26-23(20)27-25(28-24)30-9-11-34-15-17(30)2/h4-7,12,16-17,31H,8-11,13-15H2,1-3H3"
    query_name = "ku-0063794"
    query_idx = inchis == "InChI=1S/C25H31N5O4/c1-16-13-30(14-17(2)34-16)25-27-23-20(24(28-25)29-8-10-33-11-9-29)5-6-21(26-23)18-4-7-22(32-3)19(12-18)15-31/h4-7,12,16-17,31H,8-11,13-15H2,1-3H3"
    query_name = "wye-125132"
    query_idx = inchis == "InChI=1S/C27H33N7O4/c1-28-26(35)30-18-4-2-17(3-5-18)23-31-24(33-15-20-6-7-21(16-33)38-20)22-14-29-34(25(22)32-23)19-8-10-27(11-9-19)36-12-13-37-27/h2-5,14,19-21H,6-13,15-16H2,1H3,(H2,28,30,35)"
    query_name = "xl765"
    query_idx = inchis == "InChI=1S/C31H29N5O6S/c1-19-9-10-20(15-28(19)42-4)31(37)33-21-11-13-25(14-12-21)43(38,39)36-30-29(34-26-7-5-6-8-27(26)35-30)32-22-16-23(40-2)18-24(17-22)41-3/h5-18H,1-4H3,(H,32,34)(H,33,37)(H,35,36)"
    """
    query_name = "DL-ALPHA-TOCOPHEROL"
    query_idx = inchis == "InChI=1S/C29H50O2/c1-20(2)12-9-13-21(3)14-10-15-22(4)16-11-18-29(8)19-17-26-25(7)27(30)23(5)24(6)28(26)31-29/h20-22,30H,9-19H2,1-8H3"

    moa_idx = pY == moa_col  # mTOR inhibitor
    argmax = pY.argmax(1) == moa_col
    moa_preds = keys[argmax]
    moa_smiles = smiles[argmax]
    moa_conf = pY[argmax][:, moa_col]
    filt_preds, filt_smiles, filt_conf = [], [], []
    uni_preds = np.unique(moa_preds)
    for p in uni_preds:
        idx = p == moa_preds
        confs = moa_conf[idx]
        pr = moa_preds[idx]
        sm = moa_smiles[idx]
        sel = np.argmax(confs)
        filt_preds.append(pr[sel])
        filt_smiles.append(sm[sel])
        filt_conf.append(confs[sel])
    filt_preds = np.asarray(filt_preds)
    filt_smiles = np.asarray(filt_smiles)
    filt_conf = np.asarray(filt_conf)
    moa_df = pd.DataFrame(np.stack((filt_preds, filt_smiles, filt_conf), 1), columns=["keys", "smiles", "confidence"])
    files = []
    files.append("{}_predictions.csv".format(moa_name))
    moa_df.to_csv("{}_predictions.csv".format(moa_name))

    # Now filter down to compounds that we dont have annotations for
    meta = np.load("moa_metadata.npz", allow_pickle=True)
    used = meta["inchi_names"]
    sm_mask = ~np.in1d(inchis, used)
    rem_smiles = smiles[sm_mask]
    data_mask = ~np.in1d(filt_smiles, rem_smiles)
    filt_moa_df = pd.DataFrame(np.stack((filt_preds[data_mask], filt_smiles[data_mask], filt_conf[data_mask]), 1), columns=["keys", "smiles", "confidence"])
    filt_moa_df = filt_moa_df.sort_values("confidence", ascending=False)
    files.append("heldout_{}_predictions.csv".format(moa_name))
    filt_moa_df.to_csv("heldout_{}_predictions.csv".format(moa_name))

    # Make a library using nearest neighbor
    query = pX[query_idx]  # data[query_idx]
    sel_keys = keys[query_idx]
    omni_smile_key_df = ranked_library(
        query=query,
        db=pX,  # use data if you want outputs that are not aligned with MoA
        keys=keys,
        smiles=smiles,
        metric="cosine")
    clean_idx = np.in1d(omni_smile_key_df.inchi_keys.values, sel_keys)
    omni_smile_key_df = omni_smile_key_df[~clean_idx]  # Remove keys in our query
    files.append("{}_hits_v{}.csv".format(query_name, version))
    omni_smile_key_df.to_csv("{}_hits_v{}.csv".format(query_name, version))

[shutil.copy(f, "/media/data_cifs/clicktionary/clickme_experiment/tf_records/") for f in files]
print("Moved files {} to server".format(files))

