import os
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from scipy.integrate import trapz


# sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})
def get_scores(xy, tcompound_remap, target):
    score = []
    for m in xy:
        try:
            t = tcompound_remap[m]
            if target.lower() in t.lower():
                score.append(1)
            else:
                score.append(0)
        except:
            score.append(np.nan)
    score = np.asarray(score)
    return score


kind = "mol"
ddir = "embedding_data"
eval_data_dir = "eval_data"
plot_cp = False

final_data = "/media/data_cifs/projects/prj_video_imagenet/sys_linear_models/assay_data.npz"
inchi_key = "ZWYQUVBAZLHTHP-UHFFFAOYSA-N"
# InChI=1S/C56H87NO16/c1-33-17-13-12-14-18-34(2)45(68-9)29-41-22-20-39(7)56(67,73-41)51(63)52(64)57-24-16-15-19-42(57)53(65)71-46(30-43(60)35(3)26-38(6)49(62)50(70-11)48(61)37(5)25-33)36(4)27-40-21-23-44(47(28-40)69-10)72-54(66)55(8,31-58)32-59/h12-14,17-18,26,33,36-42,44-47,49-50,58-59,62,67H,15-16,19-25,27-32H2,1-11H3
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
mcrispr_X = mol_data["crispr_enc"]
mcrispr_y = mol_data["crispr_lab"]

cp_data = np.load(os.path.join(ddir, "{}_cp.npz".format(kind)))
ctrain_X = cp_data["train_enc"]
ctest_X = cp_data["test_enc"]
ctrain_y = cp_data["train_lab"]
ctest_y = cp_data["test_lab"]
ccrispr_X = cp_data["crispr_enc"]
ccrispr_y = cp_data["crispr_lab"]

# Get some info about targets for mkeeps
inv_id_remap = {v: k for k, v in id_remap.items()}
data = np.load("/media/data/final_data.npz", allow_pickle=True)
keys = data["keys"]
inchis = data["inchis"]
comps = data["compounds"]
crisprs = data["crisprs"]
# crispr_ids = np.unique(crisprs, return_inverse=True)[1]
# ccrispr_y

# Get some target annotations
td = np.load(os.path.join("eval_data", "target_data.npz"), allow_pickle=True)
# tid_remap = td["id_remap"]
tcompound_remap = td["compound_remap"].item()

train_target_names = np.load("drug_rediscov_dict.npy", allow_pickle=True).item()
inv_train_target_names = {v: k for k, v in train_target_names.items()}
rid = inv_train_target_names[inchi_key]

mu, sd = mtrain_X.mean(0), mtrain_X.std(0)
mtrain_X = (mtrain_X - mu) / sd
mtest_X = (mtest_X - mu) / sd
mcrispr_X = (mcrispr_X - mu) / sd

mu, sd = ctrain_X.mean(0), ctrain_X.std(0)
ctrain_X = (ctrain_X - mu) / sd
ctest_X = (ctest_X - mu) /sd
ccrispr_X = (ccrispr_X - mu) / sd

from scipy.spatial.distance import cdist

"""
# Make rapamycin train set
mrtr = mtrain_X[rid == mtrain_y]
crtr = ctrain_X[rid == ctrain_y]

# Make libraries
# mdm = cdist(mrtr, mtest_X, "cosine")
# cdm = cdist(crtr, ctest_X, "cosine")
mdm = cdist(mrtr, mtrain_X, "cosine")
cdm = cdist(crtr, ctrain_X, "cosine")

midx = np.argsort(mdm.min(0))
cidx = np.argsort(cdm.min(0))

my = mtrain_y[midx]
cy = ctrain_y[cidx]

# Limit to the top N
# N = 23
# top_m = [train_target_names[x] for x in my[:N]][3:]
# top_c = [train_target_names[x] for x in cy[:N]][3:]
"""

# Distance library using CRISPRs
# target = "MTOR"
target = "PAK1"
ucs = np.unique(crisprs)
ucs = ["PAK1", "DYRK1B", "TGFBR2", "CTSG", "NAMPT", "NR3C2", "MTOR", "HRH1", "ACE", "AURKC", "PTGS2", "TGFBR1"]

marea, carea = [], []
mscores, cscores = [],[]
for c in ucs:
    target_ids = np.where(crisprs == target)[0]
    sel_mcrispr_X = mcrispr_X[target_ids]
    sel_ccrispr_X = ccrispr_X[target_ids]
    # mcrdm = cdist(mtrain_X, sel_mcrispr_X, "cosine")
    # ccrdm = cdist(ctrain_X, sel_ccrispr_X, "cosine")
    mcrdm = cdist(mtest_X, sel_mcrispr_X, "cosine")
    ccrdm = cdist(ctest_X, sel_ccrispr_X, "cosine")
    mcrispr_idx = np.argsort(mcrdm.min(1))
    ccrispr_idx = np.argsort(ccrdm.min(1))
    # cmy = mtrain_y[mcrispr_idx]
    # ccy = ctrain_y[ccrispr_idx]
    cmy = mtest_y[mcrispr_idx]
    ccy = ctest_y[ccrispr_idx]

    # # Limit to the top N
    # N = 20
    # cr_top_m = [train_target_names[x] for x in cmy[:N]]
    # cr_top_c = [train_target_names[x] for x in ccy[:N]]
    mscore = get_scores(cmy, tcompound_remap, c)
    cscore = get_scores(ccy, tcompound_remap, c)
    # import pdb;pdb.set_trace()
    # plt.plot(np.arange(len(mscore)), np.cumsum(mscore), label="ours");plt.plot(np.arange(len(mscore)), np.cumsum(cscore), label="cp");plt.legend()
    ma = trapz(np.cumsum(mscore), np.arange(len(mscore)))
    ca = trapz(np.cumsum(cscore), np.arange(len(cscore)))
    # plt.show()
    marea.append(ma)
    carea.append(ca)

    mscore, cscore
    mscores.append(np.cumsum(mscore))
    cscores.append(np.cumsum(cscore))

N = 1000
f, axs = plt.subplots(1, 1)
for idx, m in enumerate(mscores):
    plt.plot(np.arange(len(m[:N])), m[:N], color="#eb4034")
for idx, m in enumerate(cscores):
    plt.plot(np.arange(len(m[:N])), m[:N], color="#348feb")
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
plt.ylim([0, np.asarray(mscores).max() + 5])
plt.savefig("results/crisprs.png", dpi=300)
plt.show()
plt.close(f)

marea = np.asarray(marea)
carea = np.asarray(carea)
stat_test = "ttest"  # "randomization"  # ttest
import pdb;pdb.set_trace()
if stat_test == "ttest":
    from scipy.stats import ttest_ind
    res = ttest_ind(marea, carea)
    p = res.p_value
elif stat_test == "randomization":
    its = 10000
    diff = marea - carea
    T = np.mean(diff)
    ts = []
    for i in range(its):
        signs = ((np.random.rand(len(marea)) - 0.5) > 0).astype(np.float32)
        ts.append(np.mean(signs * diff))
    ts = np.asarray(ts)
    p = float(1 + np.sum(ts > T)) / float(1 + its)
else:
    raise NotImplementedError
print("Stat test of IBP > CP: {}".format(p))

