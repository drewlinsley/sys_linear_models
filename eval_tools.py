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
from sklearn.metrics import average_precision_score, precision_recall_curve, f1_score
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from timm.scheduler import CosineLRScheduler
from torchsampler import ImbalancedDatasetSampler
from torcheval.metrics import MulticlassAUPRC
from torchmetrics import Accuracy


def renumber_from_0(vector):
    """Renumber a vector so that it starts from 0 and its entries are sequential."""
    unique_values = np.unique(vector)
    value_map = {value: idx for idx, value in enumerate(unique_values)}
    renumbered_vector = [value_map[value] for value in vector]
    return np.asarray(renumbered_vector)


def bootstrap(y, yhat, its, metric):
    out = []
    for i in range(its):
        idx = np.random.randint(0, len(y), size=len(y))
        it_y = y[idx]
        it_yhat = yhat[idx]
        out.append(metric(it_y, it_yhat))
    return out


class TinyModel(torch.nn.Module):

    def __init__(self, in_dims, out_dims=730, hidden=2048, linear=False):
        super(TinyModel, self).__init__()
        if linear:
            self.linear1 = torch.nn.Linear(in_dims, out_dims)  # hidden
        else:
            self.linear1 = torch.nn.Linear(in_dims, hidden)
        self.linear = linear
        self.norm1 = torch.nn.BatchNorm1d(hidden)
        self.linear2 = torch.nn.Linear(hidden, hidden)
        self.norm2 = torch.nn.BatchNorm1d(hidden)
        self.linear3 = torch.nn.Linear(hidden, out_dims)
        # self.norm3 = torch.nn.BatchNorm1d(hidden)
        # self.linear4 = torch.nn.Linear(hidden, out_dims)
        self.dropout1 = torch.nn.Dropout(0.1)
        self.dropout2 = torch.nn.Dropout(0.1)
        # self.dropout3 = torch.nn.Dropout(0.5)
        self.activation = torch.nn.GELU()

    def forward(self, x, penultimate=False):
        x = self.linear1(x)
        if self.linear:
            return x
        # x = self.norm1(x)
        x = self.dropout1(x)
        x = self.activation(x)
        x = self.linear2(x)
        # x = self.norm2(x)
        x = self.dropout2(x)
        x = self.activation(x)
        if penultimate:
            return x
        # x = self.norm2(x)
        x = self.linear3(x)
        # x = self.dropout3(x)
        # x = self.activation(x)
        # x = self.linear4(x)
        return x


def run(
        kind,
        train_X,
        train_y,
        test_X,
        test_y,
        device,
        width,
        model=None,  # Allow a model to be passed through for CP control
        crispr_data=None,
        return_data=False,
        bs=6000,  # 32768,
        test_bs=6000,
        epochs=100,
        loss_type="cce",
        lr=1e-4,
        wd=1e-6,
        rebalance=False,
        renumber=False,
        sel_train=None,
        sel_test=None,
        eval_data_dir="eval_data",
        stop_criterion=15,  # 15
        top_k=10,
        warmup_steps=200,  # 100
        warmup_epochs=40):

    """Train a readout for evaluating your model."""
    data = np.load(os.path.join(eval_data_dir, "{}_data.npz".format(kind)), allow_pickle=True)
    # train_X = data["train_X"]
    # train_y = data["train_y"]
    # test_X = data["test_X"]
    # test_y = data["test_y"]
    compound_remap = data["compound_remap"].item()
    id_remap = data["id_remap"].item()

    # Restrict to overlapping compounds
    target_keys = np.unique(np.asarray([k for k in compound_remap.keys()]))
    keep_train = np.in1d(train_y, target_keys)
    keep_test = np.in1d(test_y, target_keys)
    train_X = train_X[keep_train]
    train_y = train_y[keep_train] 
    test_X = test_X[keep_test]
    test_y = test_y[keep_test]
    if sel_train is not None:
        sel_train = sel_train[keep_train]
        sel_test = sel_test[keep_test]

    # Rebalance the categories so we have the same ones in train and test
    # One example per category
    if rebalance:
        all_X = np.concatenate((train_X, test_X), 0)
        all_y = np.concatenate((train_y, test_y), 0)
        if sel_train is not None:
            sel_all = np.concatenate((sel_train, sel_test), 0)
        uy = np.unique(all_y)
        test_idx, test_h = [], {}
        for i in range(len(all_y)):
            if all_y[i] not in test_h:
                test_idx.append(i)
                test_h[all_y[i]] = True
        test_idx = np.asarray(test_idx)
        train_idx = np.arange(len(all_y))
        train_idx = train_idx[~np.in1d(train_idx, test_idx)]
        train_X = all_X[train_idx]
        train_y = all_y[train_idx]
        test_X = all_X[test_idx]
        test_y = all_y[test_idx]
        if sel_train is not None:
            sel_train = sel_all[train_idx]
            sel_test = sel_all[test_idx]

    # Remap from compounds to test labels
    train_y = np.asarray([id_remap[compound_remap[x]] for x in train_y])
    test_y = np.asarray([id_remap[compound_remap[x]] for x in test_y])

    if renumber:
        all_y = np.concatenate((train_y, test_y))
        idx = np.concatenate((np.zeros((len(train_y))), np.ones((len(test_y)))))
        all_y = renumber_from_0(all_y)
        train_y = all_y[idx == 0]
        test_y = all_y[idx == 1]

    # Hyperparam s
    nc = int(train_y.max() + 1)  # len(np.unique(train_y))  # torch.unique(train_y))
    if model is None:
        model = TinyModel(in_dims=width, out_dims=nc).to("cuda")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        weight_decay=wd,
        lr=lr)

    # Prepare loss
    if loss_type == "cce":
        loss = torch.nn.CrossEntropyLoss()
    elif loss_type == "bce":
        loss = torch.nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError(loss_type)

    # Prepare data
    pt_train_X = torch.from_numpy(train_X).float()
    pt_train_Y = torch.from_numpy(train_y).long()
    pt_test_X = torch.from_numpy(test_X).float()
    pt_test_Y = torch.from_numpy(test_y).long()

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
    if crispr_data is not None:
        pt_crispr_X = torch.from_numpy(crispr_data).float()
        crispr_dataset = torch.utils.data.TensorDataset(
            pt_crispr_X)
        
    # sampler = ImbalancedDatasetSampler(train_dataset)
    bs = min(bs, len(train_dataset))
    test_bs = min(test_bs, len(test_dataset))
    print("For task: {} train samples: {} test samples: {}".format(kind, len(train_dataset), len(test_dataset)))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        drop_last=True,
        batch_size=bs,
        shuffle=True)
        # sampler=sampler)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_bs,  # len(test_dataset),
        drop_last=False,
        shuffle=False)
    # scheduler = get_linear_schedule_with_warmup(  # get_cosine_schedule_with_warmup(  # get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=warmup_steps,
    #     num_training_steps=epochs * int(len(train_loader) // bs)
    # )
    # scheduler = CosineLRScheduler(
    #     optimizer,
    #     t_initial=epochs,
    #     warmup_t=10,
    #     warmup_lr_init=1e-6)

    # Run training
    n_b = len(train_dataset) // bs
    best_loss = 100
    best_acc = 0
    for epoch in range(epochs):
        progress = tqdm(total=1, desc="Training")
        model.train()
        losses = []
        accs = []
        # metric = MulticlassAUPRC(num_classes=nc)
        metric = Accuracy(task="multiclass", num_classes=nc, top_k=top_k)
        for batch in train_loader:
            it_X, it_y = batch
            it_X = it_X.to(device)
            it_y = it_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            z = model(it_X)
            if loss_type == "bce":
                it_y = torch.nn.functional.one_hot(it_y, nc).float()
            l = loss(z, it_y)

            # z_q = (z > 0.5).float().cpu().numpy()
            # z_q = z.argmax(1).detach().cpu()
            # acc = bas(it_y.cpu().numpy(), z_q)
            # metric.update(z, it_y)
            accs.append(metric(z.cpu(), it_y.cpu()))

            l.backward()
            optimizer.step()
            # scheduler.step()  # Let's go without the scheduler for now
            losses.append(l.item())
        # acc = metric.compute()
        acc = np.mean(accs)
        progress.set_postfix(
            {
                "train_loss": np.mean(losses),
                "train_acc": acc})
        progress.update()

        model.eval()
        test_loss, preds, gts = [], [], []
        # metric = MulticlassAUPRC(num_classes=nc)
        metric = Accuracy(task="multiclass", num_classes=nc, top_k=top_k)
        accs = []
        for batch in test_loader:
            with torch.no_grad():
                it_X, it_y = batch
                it_X = it_X.to(device)
                it_y = it_y.to(device)
                z = model(it_X)
                if loss_type == "bce":
                    it_y = torch.nn.functional.one_hot(it_y, nc).float()
                l = loss(z, it_y)
                # z_q = (z > 0.5).float().cpu().numpy()
                # z_q = z.argmax(1).detach().cpu().numpy()
                # acc = bas(it_y.cpu().numpy(), z_q)
                # preds.append(z_q)
                # preds.append(z)  # .detach().cpu().numpy())
                # gts.append(it_y)  # .cpu().numpy())
                l = l.item()
                test_loss.append(l)
                # metric.update(z, it_y)
                accs.append(metric(z.cpu(), it_y.cpu()))
        it_test_loss = np.mean(test_loss)
        # it_best_acc = metric.compute()
        it_best_acc = np.mean(accs)
        progress.set_postfix(
            {
                # "lr": scheduler.get_last_lr()[0],
                "test_loss": it_test_loss,
                "test_acc": it_best_acc},
            )
        progress.update()
        progress.close()

        # Save best model
        # if it_test_loss < best_loss:
        if it_best_acc > best_acc:
            # best_loss = it_test_loss
            best_acc = it_best_acc
            best_loss = it_test_loss
            epoch_counter = 0
            loss_std = np.std(test_loss) / np.sqrt(len(test_loss))

            # Compute p/r
            # f1_score(np.concatenate(gts), torch.cat(preds), average="weighted")
            # cat_gts = np.concatenate(gts)
            # cat_preds = np.concatenate(preds, 0)
            # sim_ap = bootstrap(y=cat_gts, yhat=cat_preds, its=1000, metric=lambda x,y: average_precision_score(x, y, average="micro"))
            # acc_std = np.std(sim_ap) / np.sqrt(len(sim_ap))
            acc_std = 0.
            print("Updated best score.")
        else:
            if epoch > warmup_epochs:  # Start counting for early stopping
                epoch_counter += 1

        if epoch_counter > stop_criterion:
            print("Triggered early stopping.")
            break  # Early stopping triggered
    results = {
        "acc": best_acc.item(),
        "acc_std": acc_std,
        "loss": best_loss,
        "loss_std": loss_std, 
    }
    if kind == "moa" and sel_train is not None:
        # Add some rediscovery results
        # Encode training set
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=bs,
            drop_last=False,
            shuffle=False)
        train_enc, train_lab = [], []
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc="Processing training data"):
                it_X, it_y = batch
                it_X = it_X.to(device)
                it_y = it_y.to(device)
                z = model(it_X)
                train_enc.append(z)
                train_lab.append(it_y)
        train_enc = torch.cat(train_enc).cpu().numpy()
        train_lab = torch.cat(train_lab).cpu().numpy()

        # Encode test set
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=bs,
            drop_last=False,
            shuffle=False)
        test_enc, test_lab = [], []
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(test_loader), total=len(train_loader), desc="Processing training data"):
                it_X, it_y = batch
                it_X = it_X.to(device)
                it_y = it_y.to(device)
                z = model(it_X)
                test_enc.append(z)
                test_lab.append(it_y)
        test_enc = torch.cat(test_enc).cpu().numpy()  # Slow but convenient. Consider removing.
        test_lab = torch.cat(test_lab).cpu().numpy()

        # How close is Rapamaycin to Rapalogues (other mTOR inhibitors)
        redisc_train_X = train_enc[sel_train]
        redisc_test_X = test_enc[sel_test]
        dists_tr = cdist(redisc_train_X, test_enc, metric="cosine")
        dists_te = cdist(redisc_test_X, test_enc, metric="cosine")
        dists = np.concatenate((dists_tr, dists_te), 0).mean(0)
        sims = 1 - dists  # Convert to similarities for z-scores below
        ks = np.asarray([x for x in id_remap.keys()])
        rapa_id = np.where(np.logical_or(ks == "mTOR inhibitor", ks == "mTOR inhibitor|PI3K inhibitor"))[0]
        # cont_id = np.where(ks == "contrast agent")[0][0]
        ridx = np.in1d(test_y, rapa_id)
        nonridx = np.roll(ridx, ridx.sum())  # Roll the index by the number of rapalogues
        rapa_sims = sims[ridx]  # Rapamycin vs. rapalogue similarity
        cont_sims = sims[nonridx]  # Rapamycin vs. other similarity
        raw_dist = rapa_sims.mean() - cont_sims.mean()  # Aiming for positive large values
        norm_s = (rapa_sims.mean() / rapa_sims.std()) - (cont_sims.mean() / cont_sims.std())
        results["rediscovery_acc"] = norm_s
        results["rediscovery_z"] = raw_dist

    if return_data:
        model.eval()
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=bs,
            drop_last=False,
            shuffle=False)
        train_enc, train_lab = [], []
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc="Processing training data"):
                it_X, it_y = batch
                it_X = it_X.to(device)
                it_y = it_y.to(device)
                z = model(it_X, penultimate=True)
                train_enc.append(z)
                train_lab.append(it_y)
        train_enc = torch.cat(train_enc).cpu().numpy()
        train_lab = torch.cat(train_lab).cpu().numpy()

        # Encode test set
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=bs,
            drop_last=False,
            shuffle=False)
        test_enc, test_lab = [], []
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc="Processing testing data"):
                it_X, it_y = batch
                it_X = it_X.to(device)
                it_y = it_y.to(device)
                z = model(it_X, penultimate=True)
                test_enc.append(z)
                test_lab.append(it_y)
        test_enc = torch.cat(test_enc).cpu().numpy()  # Slow but convenient. Consider removing.
        test_lab = torch.cat(test_lab).cpu().numpy()
        if crispr_data is not None:
            crispr_loader = torch.utils.data.DataLoader(
                crispr_dataset,
                batch_size=bs,
                drop_last=False,
                shuffle=False)
            crispr_enc = []
            with torch.no_grad():
                for batch_idx, batch in tqdm(enumerate(crispr_loader), total=len(crispr_loader), desc="Processing crispr data"):
                    it_X = batch[0]
                    it_X = it_X.to(device)
                    z = model(it_X, penultimate=True)
                    crispr_enc.append(z)
            crispr_enc = torch.cat(crispr_enc).cpu().numpy()  # Slow but convenient. Consider removing.
            return results, train_enc, test_enc, train_lab, test_lab, crispr_enc
        else:
            return results, train_enc, test_enc, train_lab, test_lab
    else:
        return results


if __name__ == '__main__':
    main()

