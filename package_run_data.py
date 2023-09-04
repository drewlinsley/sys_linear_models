import os
import numpy as np


perturb_data="/media/data/final_data.npz"
eval_data_dir = "eval_data"
moa_eval_data = np.load(os.path.join(eval_data_dir, "{}_data.npz".format("moa")), allow_pickle=True)
target_eval_data = np.load(os.path.join(eval_data_dir, "{}_data.npz".format("target")), allow_pickle=True)
moa_compound_remap = moa_eval_data["compound_remap"].item()
target_compound_remap = target_eval_data["compound_remap"].item()
moa_compounds = [k for k in moa_compound_remap.keys()]
target_compounds = [k for k in target_compound_remap.keys()]
all_compounds_to_keep = np.unique(moa_compounds + target_compounds)


data = np.load(perturb_data, allow_pickle=True)
train_res, train_source, train_batch, train_well, train_compounds = data["train_res"], data["train_source"], data["train_batch"], data["train_well"], data["train_compounds"]
test_res, test_source, test_batch, test_well, test_compounds = data["test_res"], data["test_source"], data["test_batch"], data["test_well"], data["test_compounds"]
keys, compounds = data["keys"], data["compounds"]
keep_key = "ZWYQUVBAZLHTHP-UHFFFAOYSA-N"

all_res = np.concatenate((train_res, test_res), 0)
all_res = np.clip(all_res, -30, 30)
all_source = np.concatenate((train_source, test_source), 0)
all_batch = np.concatenate((train_batch, test_batch), 0)
all_well = np.concatenate((train_well, test_well), 0)
all_comp = np.concatenate((train_compounds, test_compounds), 0)
# ucomp = np.unique(all_comp)

test_exemplars = 0.01  # This prportion held out for testing
h = {k: 0 for k in all_compounds_to_keep}

# Find rapamaycin make sure it's included
h[compounds[keys == keep_key][0]] = 0

# Get counts for each compound
hc = {}
for c in all_compounds_to_keep:
    thresh = np.ceil((all_comp == c).sum() * test_exemplars)
    if thresh < 1:
        import pdb;pdb.set_trace()
    hc[c] = thresh

test_idx = []
for idx, c in enumerate(all_comp):
   if c in h and h[c] <= hc[c]:
      h[c] += 1
      test_idx.append(True)
   else:
      test_idx.append(False)
test_idx = np.asarray(test_idx)
train_res, test_res = all_res[~test_idx], all_res[test_idx]
print("Train samples: {}".format(len(train_res)))
print("Test samples: {}".format(len(test_res)))
# mu, sd = train_res.mean(0), train_res.std(0)
# train_res = (train_res - mu) / sd
# test_res = (test_res - mu) / sd

train_source, test_source = all_source[~test_idx], all_source[test_idx]
train_well, test_well = all_well[~test_idx], all_well[test_idx]
train_batch, test_batch = all_batch[~test_idx], all_batch[test_idx]
train_compounds, test_compounds = all_comp[~test_idx], all_comp[test_idx]
np.savez("/media/data_cifs/projects/prj_video_imagenet/sys_linear_models/assay_data.npz", train_batch=train_batch, test_batch=test_batch, train_res=train_res, test_res=test_res, train_source=train_source, test_source=test_source, train_well=train_well, test_well=test_well, train_compounds=train_compounds, test_compounds=test_compounds, test_idx=test_idx, keys=keys, compounds=compounds)

