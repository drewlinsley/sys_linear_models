import os
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import matplotlib as mpl


# sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})


kind = "target"
ddir = "embedding_data"
eval_data_dir = "eval_data"
plot_cp = False

final_data = "/media/data_cifs/projects/prj_video_imagenet/sys_linear_models/assay_data.npz"
inchi_key = "ZWYQUVBAZLHTHP-UHFFFAOYSA-N"
if kind == "mol":
    data = np.load(os.path.join(eval_data_dir, "target_data.npz"), allow_pickle=True)
else:
    data = np.load(os.path.join(eval_data_dir, "{}_data.npz".format(kind)), allow_pickle=True)
compound_remap = data["compound_remap"].item()
id_remap = data["id_remap"].item()

if kind =="target":
    thresh = 25
elif kind == "moa":
    thresh = 75
elif kind == "mol":
    thresh = 4
else:
    raise NotImplementedError

mol_data = np.load(os.path.join(ddir, "{}_best_mol.npz".format(kind)))
mtrain_X = mol_data["train_enc"]
mtest_X = mol_data["test_enc"]
mtrain_y = mol_data["train_lab"]
mtest_y = mol_data["test_lab"]

f_data = np.load(os.path.join(ddir, "{}_50_mol.npz".format(kind)))
ftrain_X = f_data["train_enc"]
ftest_X = f_data["test_enc"]
ftrain_y = f_data["train_lab"]
ftest_y = f_data["test_lab"]

cp_data = np.load(os.path.join(ddir, "{}_cp.npz".format(kind)))
ctrain_X = cp_data["train_enc"]
ctest_X = cp_data["test_enc"]
ctrain_y = cp_data["train_lab"]
ctest_y = cp_data["test_lab"]

"""
data = np.load(final_data, allow_pickle=True)
keys, compounds = data["keys"], data["compounds"]
train_compounds = data["train_compounds"]
test_compounds = data["test_compounds"]
sel_keys = keys == inchi_key
sel_comps = compounds[sel_keys]
# sel_train = train_compounds == np.unique(sel_comps).squeeze()
# sel_test = test_compounds == np.unique(sel_comps).squeeze()
"""

# Restrict to overlapping compounds
"""
target_keys = np.unique(np.asarray([k for k in compound_remap.keys()]))
keep_train = np.in1d(train_y, target_keys)
keep_test = np.in1d(test_y, target_keys)
mtrain_X = mtrain_X[keep_train]
train_y = train_y[keep_train]
mtest_X = mtest_X[keep_test]
test_y = test_y[keep_test]
# sel_train = sel_train[keep_train]
# sel_test = sel_test[keep_test]
train_y = np.asarray([id_remap[compound_remap[x]] for x in train_y])
test_y = np.asarray([id_remap[compound_remap[x]] for x in test_y])
"""

# csort = np.argsort(train_y)
# ssort = np.argsort(test_y)

# 14, 12, 22
mkeeps = np.unique(mtest_y, return_counts=True)[1]
mkeeps = np.where(np.logical_and(mkeeps > thresh, mkeeps < mkeeps.max()))[0]  # np.sort(np.unique(mtest_y, return_counts=True)[1])[-10:]
mkeeps = np.unique(mtest_y)[mkeeps]
# ckeeps = np.unique(ctest_y)[ckeeps]
ckeeps = mkeeps

# Get some info about targets for mkeeps
inv_id_remap = {v: k for k, v in id_remap.items()}
if kind == "mol":
    data = np.load("/media/data/final_data.npz", allow_pickle=True)
    keys = data["keys"]
    comps = data["compounds"]
    target_names = []
    for m in mtest_y:
        ik = keys[comps == m]
        target_names.append(ik[0])
    target_names = np.asarray(target_names)
    target_names = {m: t for m, t in zip(mtest_y, target_names)}
else:
    target_names = {x: inv_id_remap[x] for x in mkeeps}
    target_names = {k: v.split("|")[-1] for k, v in target_names.items()}

# mkeeps = np.sort(np.unique(mtest_y, return_counts=True)[1])[-5:]

# mkeeps = np.unique(mtest_y)[np.argsort(np.unique(mtest_y, return_counts=True)[1])[-35:-5]]
# ckeeps = np.unique(mtest_y)[np.argsort(np.unique(ctest_y, return_counts=True)[1])[-35:-5]]
tr_csort = np.in1d(ctrain_y, ckeeps)
te_csort = np.in1d(ctest_y, ckeeps)

tr_msort = np.in1d(mtrain_y, mkeeps)
te_msort = np.in1d(mtest_y, mkeeps)

#mtrain_X = mtrain_X[tr_msort]
mtest_X = mtest_X[te_msort]
#ctrain_X = ctrain_X[tr_csort]
ctest_X = ctest_X[te_csort]
#mtrain_y = mtrain_y[tr_msort]
mtest_y = mtest_y[te_msort]
#ctrain_y = ctrain_y[tr_csort]
ctest_y = ctest_y[te_csort]
ftest_X = ftest_X[te_csort]
msc = StandardScaler().fit(mtrain_X)
mtrain_X = msc.transform(mtrain_X)
mtest_X = msc.transform(mtest_X)

csc = StandardScaler().fit(ctrain_X)
ctrain_X = csc.transform(ctrain_X)
ctest_X = csc.transform(ctest_X)

# fsc = StandardScaler().fit(ftrain_X)
# ftrain_X = csc.transform(ftrain_X)
# ftest_X = csc.transform(ftest_X)

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
reducer = umap.UMAP(random_state=42, n_neighbors=50, min_dist=0.7)
embedding = reducer.fit_transform(mtest_X)
# fembedding = reducer.fit_transform(ftest_X)
reducer = umap.UMAP(random_state=42, n_neighbors=30, min_dist=0.1)
cembedding = reducer.fit_transform(ctest_X)
# clf = PCA(whiten=False, n_components=2).fit(mtrain_X)
# embedding = clf.transform(mtest_X)

# clf = PCA(whiten=True, n_components=2).fit(ftrain_X)
# fembedding = clf.transform(ftest_X)

# clf = PCA(whiten=False, n_components=2).fit(ctrain_X)
# cembedding = clf.transform(ctest_X)
# embedding = TSNE(perplexity=50, n_components=2).fit_transform(mtest_X)
# fembedding = TSNE(perplexity=50, n_components=2).fit_transform(ftest_X)
# cembedding = TSNE(perplexity=50, n_components=2).fit_transform(ctest_X)

# Get means
means = {}
for k in target_names.keys():
    idx = mtest_y == k
    means[k] = embedding[idx].mean(0)

cmap_n = "rainbow"
# cmap_n = "nipy_spectral"
# cmap_n = "inferno"
cmap_n = "turbo"
# cmap_n = "gist_ncar"
# cmap_n = "tab20"
# cmap_n = "Paired"

cmap = sns.color_palette(cmap_n, len(np.unique(np.unique(mtest_y))))
uy = np.unique(mtest_y)

import matplotlib.colors as colors
if plot_cp:
    f = plt.figure()
    ax = plt.subplot(111)
    for idx, i in enumerate(uy):
        inner_c = mpl.colors.colorConverter.to_rgba(cmap[idx], alpha=.5)
        # plt.plot(cembedding[mtest_y == i, 0], cembedding[mtest_y == i, 1], ".", markersize=5, alpha=0.9, c=inner_c, markeredgecolor=cmap[idx])
        plt.scatter(cembedding[mtest_y == i, 0], cembedding[mtest_y == i, 1], s=5, c=inner_c, edgecolor=cmap[idx], alpha=0.8)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_aspect(1)
    plt.title("Cellprofiler")
    plt.show()
    plt.close("all")

f = plt.figure()
ax = plt.subplot(111)
sh = np.random.permutation(len(uy))
for idx, i in enumerate(uy):
    idx = sh[idx]
    # inner_c = mpl.colors.colorConverter.to_rgba(cmap[idx], alpha=.5)
    # plt.scatter(embedding[mtest_y == i, 0], embedding[mtest_y == i, 1], s=10, c=inner_c, edgecolor=cmap[idx], linewidth=2)
    plt.scatter(embedding[mtest_y == i, 0], embedding[mtest_y == i, 1], s=20, color=cmap[idx], alpha=0.8)
    mu = means[i]
    txt = target_names[i]
    # plt.text(mu[0], mu[1], txt)  # , fontsize=16, fontweight="bold")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# ax.set_aspect(1)
plt.title("Optimal ours")
plt.savefig(os.path.join("results", "{}_embedding.png".format(kind)), dpi=600)
# plt.show()
plt.close(f)

f = plt.figure()
ax = plt.subplot(111)
sh = np.random.permutation(len(uy))
for idx, i in enumerate(uy):
    idx = sh[idx]
    inner_c = mpl.colors.colorConverter.to_rgba(cmap[idx], alpha=.5)
    # plt.scatter(embedding[mtest_y == i, 0], embedding[mtest_y == i, 1], s=5, c=inner_c, edgecolor=cmap[idx], linewidth=2)
    plt.scatter(embedding[mtest_y == i, 0], embedding[mtest_y == i, 1], s=10, color=cmap[idx], alpha=0.5)
    mu = means[i]
    txt = target_names[i]
    plt.text(mu[0], mu[1], txt)  # , fontsize=16, fontweight="bold")
    print(txt)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_aspect(1)
plt.title("Optimal ours")
plt.savefig(os.path.join("results", "{}_text_embedding.png".format(kind)), dpi=600)
# plt.show()
plt.close(f)

os._exit(1)

####

f = plt.figure()

plt.subplot(131)
plt.scatter(cembedding[:, 0], cembedding[:, 1], c=ctest_y, s=10, cmap=cmap)
# for i in uy:
#     plt.plot(cembedding[test_y == i, 0], cembedding[test_y == i, 1], ".", label=i)
plt.title("Cell profiler")


plt.subplot(132)
plt.scatter(fembedding[:, 0], fembedding[:, 1], c=mtest_y, s=10, cmap=cmap)
# uy = np.unique(test_y)[:50]
# for i in uy:
#     plt.plot(embedding[test_y == i, 0], embedding[test_y == i, 1], ".", label=i)
plt.title("50 Molecule pretraining")

plt.subplot(133)
plt.scatter(embedding[:, 0], embedding[:, 1], c=mtest_y, s=10, cmap=cmap)
# uy = np.unique(test_y)[:50]
# for i in uy:
#     plt.plot(embedding[test_y == i, 0], embedding[test_y == i, 1], ".", label=i)
plt.title("100 Molecule pretraining")


plt.show()

