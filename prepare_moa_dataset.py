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
    for k, s in zip(train_key, train_smile):
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

# Make mapping between keys and inchis
inv_ck_map = {k: v for k, v in zip(keys, inchis)}

# Load and combine repurposing spreadsheets
df = pd.read_csv("repurpose_database_X_jump.csv")
rephub = pd.read_csv("compound_rephub_annot_full.csv.gz")
antiox = pd.read_csv("antioxidants.csv")

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
for i, t in tqdm(enumerate(umoas), total=umoa_len, desc="MoAs"):
    ikeys = df_comb[moas == t].inchi_key.values
    idx = np.in1d(keys, ikeys)
    act = data[idx]
    cont_act = control_data[idx]
    ikeys = keys[idx]
    iinchi = inchis[idx]
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
remap = {c: idx for idx, c in enumerate(uni_cats)}
train_y = np.asarray([remap[x] for x in train_y])
test_y = np.asarray([remap[x] for x in test_y])

# Save metadata
np.savez("moa_metadata", inchi_names=np.unique(train_inchi))

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
            # self.linear4 = torch.nn.Linear(2048, 2048)
            # self.linear5 = torch.nn.Linear(2048, out_dims)
            # # self.out = torch.nn.Sigmoid()
            self.activation = torch.nn.GELU()
            # self.norm = torch.nn.InstanceNorm1d(2048)
            # self.linear2 = torch.nn.Linear(200, 10)
            # self.softmax = torch.nn.Softmax()

        def forward(self, x):
            x = self.linear1(x)
            # x = self.norm(x)
            x = self.dropout1(x)
            x = self.activation(x)
            x = self.linear2(x)
            x = self.dropout2(x)
            # x = self.norm(x)
            x = self.activation(x)
            x = self.linear3(x)
            # x = self.activation(x)
            # x = self.linear4(x)
            # x = self.activation(x)
            # x = self.linear5(x)

            # x = self.softmax(x)
            # return self.out(x)
            return x


    bs = 32768
    nc = len(np.unique(train_y))
    epochs = 100
    loss_type = "cce"
    lr = 1e-4
    wd = 1e-4
    model = TinyModel(out_dims=nc).to("cuda")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        weight_decay=wd,
        lr=lr)  # ,

    # Prepare loss
    if loss_type == "cce":
        loss = torch.nn.CrossEntropyLoss()
    elif loss_type == "bce":
        loss = torch.nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError(loss_type)

    # Prepare data
    pt_train_X = torch.from_numpy(train_X).float().cuda()
    pt_train_Y = torch.from_numpy(np.unique(train_y, return_inverse=True)[1]).long().cuda()
    pt_test_X = torch.from_numpy(test_X).float().cuda()
    pt_test_Y = torch.from_numpy(np.unique(test_yi, return_inverse=True)[1]).long().cuda()

    # Make sampler with inverse weighting
    print("Building sampler")
    uni_c, class_sample_count = np.unique(train_y, return_counts=True)
    weight = 1. / class_sample_count
    weight_dict = {k: v for k, v in zip(uni_c, weight)}
    samples_weight = np.array([weight_dict[t] for t in train_y])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
    print("Sampler built")

    # Make data loaders
    train_dataset = torch.utils.data.TensorDataset(
        pt_train_X,
        pt_train_Y)
    test_dataset = torch.utils.data.TensorDataset(
        pt_test_X,
        pt_test_Y)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        drop_last=True,
        batch_size=bs,
        sampler=sampler)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=len(test_dataset),
        drop_last=True,
        shuffle=False)

    # Run training
    n_b = len(train_dataset) // bs
    best_loss = 100
    for epoch in range(epochs):
        progress = tqdm(total=n_b, desc="Training")
        model.train()
        for batch in train_loader:
            it_X, it_y = batch

            optimizer.zero_grad(set_to_none=True)
            z = model(it_X)
            if loss_type == "bce":
                it_y = torch.nn.functional.one_hot(it_y, nc).float()
            l = loss(z, it_y)

            # z_q = (z > 0.5).float().cpu().numpy()
            z_q = z.argmax(1).detach().cpu()
            acc = bas(it_y.cpu().numpy(), z_q)

            l.backward()
            optimizer.step()
            progress.set_postfix(
                {
                    "train_loss": l.item(),
                    "train_acc": acc})
            progress.update()

        model.eval()
        test_loss = []
        for batch in test_loader:
            with torch.no_grad():
                it_X, it_y = batch
                z = model(it_X)
                if loss_type == "bce":
                    it_y = torch.nn.functional.one_hot(it_y, nc).float()
                l = loss(z, it_y)
                # z_q = (z > 0.5).float().cpu().numpy()
                z_q = z.argmax(1).detach().cpu()
                acc = bas(it_y.cpu().numpy(), z_q)
                l = l.item()
                progress.set_postfix(
                    {
                        "test_loss": l,
                        "test_acc": acc})
                progress.update()
                test_loss.append(l)
        progress.close()

        # Save best model
        test_loss = np.mean(test_loss)
        if test_loss < best_loss:
            print("Saving")
            torch.save(model.state_dict(), out_path)

