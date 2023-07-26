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


class TinyModel(torch.nn.Module):

    def __init__(self, out_dims=730, hidden=2048):
        super(TinyModel, self).__init__()
        self.linear1 = torch.nn.Linear(1024, hidden)
        self.linear2 = torch.nn.Linear(hidden, hidden)
        self.linear3 = torch.nn.Linear(hidden, out_dims)
        self.dropout1 = torch.nn.Dropout(0.1)
        self.dropout2 = torch.nn.Dropout(0.1)
        self.activation = torch.nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        x = self.activation(x)
        x = self.linear3(x)
        return x


def run(kind, bs=32768, epochs=100, loss_type="cce", lr=1e-4, wd=1e-4, eval_data_dir="eval_data"):

    # Train models
    data = np.load(os.path.join(eval_data_dir, "{}_data.npz".format(kind)))
    train_X = data["train_X"]
    train_y = data["train_y"]
    test_X = data["test_X"]
    test_y = data["test_y"]

    # Hyperparams
    nc = len(np.unique(train_y))
    model = TinyModel(out_dims=nc).to("cuda")
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
    pt_train_X = torch.from_numpy(train_X).float().cuda()
    pt_train_Y = torch.from_numpy(np.unique(train_y, return_inverse=True)[1]).long().cuda()
    pt_test_X = torch.from_numpy(test_X).float().cuda()
    pt_test_Y = torch.from_numpy(np.unique(test_y, return_inverse=True)[1]).long().cuda()

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
        test_loss, accs = [], []
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
                accs.append(acc)
        progress.close()

        # Save best model
        it_test_loss = np.mean(test_loss)
        if it_test_loss < best_loss:
            best_loss = it_test_loss
            loss_std = np.std(test_loss)
            best_acc = np.mean(accs)
            acc_std = np.std(accs)
    results = {
        "acc": best_acc,
        "acc_std": acc_std,
        "loss": best_loss,
        "loss_std": loss_std, 
    }
    return results


if __name__ == '__main__':
    main()

