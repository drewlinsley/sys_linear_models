import os
import sys
import random

import numpy as np
import pandas as pd

from functools import partial

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.utils.data.sampler import WeightedRandomSampler

from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
# from accelerate import Accelerator

import db_utils
import eval_tools

from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)


class Mmd_resnet(nn.Module):   
    def __init__(self, 
                 input_dim,
                 int_dim,
                 output_dim,
                 n_blocks,
                 num_embeddings_b,
                 num_embeddings_s,
                 # num_embeddings_p,
                 num_embeddings_w,
                 embedding_dim_b,
                 embedding_dim_s,
                 # embedding_dim_p,
                 embedding_dim_w,
                 norm_type=nn.BatchNorm1d,  # nn.LayerNorm,  # torch.nn.Identity,  # torch.nn.BatchNorm1d,
                 use_dropout=0.1):
        super(Mmd_resnet, self).__init__()    

        self.n_blocks = n_blocks

        # Make IV and DV networks
        self.embedding_s = nn.Sequential(*[
            torch.nn.Embedding(
                num_embeddings=num_embeddings_s,
                embedding_dim=embedding_dim_s),
            torch.nn.Linear(embedding_dim_s, embedding_dim_s),
            norm_type(embedding_dim_s)
        ])
        self.proj = nn.Sequential(*[
            nn.Linear(input_dim, int_dim),
            norm_type(int_dim)
        ])

        self.iv_layers, self.dv_layers = [], []
        for l in range(self.n_blocks):
            self.dv_layers.append(nn.Sequential(*[
                torch.nn.Linear(int_dim + embedding_dim_s, int_dim),
                torch.nn.Dropout(use_dropout),
                torch.nn.GELU(),
                norm_type(int_dim),  # BatchNorm1d(dim),
            ]))
        self.dv_layers = nn.ModuleList(self.dv_layers)
        self.final = nn.Sequential(*[
            torch.nn.Linear(int_dim, output_dim)
        ])
        self.b = nn.Linear(int_dim, num_embeddings_b)
        self.s = nn.Linear(int_dim, num_embeddings_s)
        self.w = nn.Linear(int_dim, num_embeddings_w)
        
    def forward(self, dv, iv_s, iv_b, iv_w, return_p = False):
        """Forward function (with skip connections)"""
        y = self.proj(dv)
        x_s = self.embedding_s(iv_s).squeeze(1)
        for l in range(self.n_blocks):
            dv_layer = self.dv_layers[l]
            cat_y = torch.concat((y, x_s), 1)
            if l % 2:
                y = dv_layer(cat_y) + y
            else:
                y = dv_layer(cat_y)  # Skip the nonlinear layer
        out = self.final(y)
        b = self.b(y)
        s = self.s(y)
        w = self.w(y)
        if return_p:
            return out, y
        else:
            return out, b, s, w

     
def main(
        id,
        data_prop,
        label_prop,
        objective,
        lr,
        bs,
        moa,
        target,
        layers,
        width,
        batch_effect_correct,

        # These are not used but come from DB
        reserved=None,
        finished=None,

        # Defaults below are fixed
        test_epochs=100,
        version=24,
        final_data="/media/data/final_data.npz",
        ckpt_dir="sys_ckpts",
    ):
    """Run one iteration of training and evaluation."""
    # accelerator = Accelerator()
    # device = accelerator.device
    device = "cuda"

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
    out_name = "data_{}_model_{}.pth".format(version, id)
    path = os.path.join(ckpt_dir, out_name)

    # Load data
    data = np.load(final_data)
    train_res, train_source, train_batch, train_well, train_compounds = data["train_res"], data["train_source"], data["train_batch"], data["train_well"], data["train_compounds"]
    test_res, test_source, test_batch, test_well, test_compounds = data["test_res"], data["test_source"], data["test_batch"], data["test_well"], data["test_compounds"]
    orf_data, orf_source, orf_batch, orf_well = data["orf_data"], data["orf_source"], data["orf_batch"], data["orf_well"]
    crispr_data, crispr_source, crispr_batch, crispr_well = data["orf_data"], data["orf_source"], data["orf_batch"], data["orf_well"]
    res, orfs, crisprs = data["res"], data["orfs"], data["crisprs"]

    # Default params
    emb_dim_b = 64
    emb_dim_s = 16  # 4
    emb_dim_p = 16
    emb_dim_w = 16
    num_embeddings_b = train_batch.max() + 1  # train_batch.shape[-1]
    num_embeddings_s = train_source.max() + 1  # train_source.shape[-1]
    num_embeddings_w = train_well.max() + 1  # train_well.shape[-1]
    input_dim = train_res.shape[-1]
    output_dim = train_compounds.max() + 1

    # Counting variables for training
    epochs = 2000
    best_loss = 10000000
    epoch_counter = 0
    balanced_loss = False
    int_dim = 1024  # 600 # input_dim + emb_dim
    best_test_acc = 0.
    eb = None
    teb = None
    tops = 10  # top-K classification accuracy
    nc = len(np.unique(train_compounds))

    # Inverse weighting for sampling
    uni_c, class_sample_count = np.unique(train_compounds, return_counts=True)
    weight = 1. / class_sample_count
    weight_dict = {k: v for k, v in zip(uni_c, weight)}
    samples_weight = np.array([weight_dict[t] for t in train_compounds])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    # Build data
    train_res = torch.Tensor(train_res).float()
    train_source = torch.Tensor(train_source).long()
    train_batch = torch.Tensor(train_batch).long()
    train_well = torch.Tensor(train_well).long()
    train_compounds = torch.Tensor(train_compounds).long()
    test_res = torch.Tensor(test_res).float()
    test_source = torch.Tensor(test_source).long()
    test_batch = torch.Tensor(test_batch).long()
    test_well = torch.Tensor(test_well).long()
    test_compounds = torch.Tensor(test_compounds).long()
    train_dataset = torch.utils.data.TensorDataset(
        train_res,
        train_compounds,
        train_source,
        train_batch,
        train_well)
    test_dataset = torch.utils.data.TensorDataset(
        test_res,
        test_compounds,
        test_source,
        test_batch,
        test_well)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        drop_last=True,
        batch_size=bs,
        sampler=sampler,
        pin_memory=True)  # Remove pin memory if using accelerate
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=34,
        drop_last=True,
        shuffle=False,
        pin_memory=True)

    # Build model etc
    model = Mmd_resnet(
        input_dim,
        int_dim,
        output_dim,
        layers,
        num_embeddings_b=num_embeddings_b,
        num_embeddings_s=num_embeddings_s,
        num_embeddings_w=num_embeddings_w,
        embedding_dim_b=emb_dim_b,
        embedding_dim_s=emb_dim_s,
        embedding_dim_w=emb_dim_w)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        weight_decay=1e-6,  # Default
        lr=lr)  # ,
    scheduler = get_cosine_schedule_with_warmup(  # get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=500,
        num_training_steps=epochs * int(len(train_loader) // bs)
    )
    # model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
    #     model,
    #     optimizer,
    #     train_loader,
    #     test_loader,
    #     scheduler)
    # model, optimizer, train_loader, test_loader, orf_loader, crispr_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, test_loader, orf_loader, crispr_loader, scheduler)
    model.to(device)
    avg_loss = torch.tensor(0).float().to(device)

    # accelerator.wait_for_everyone()
    for epoch in range(epochs):
        batch_losses = []
        # progress = tqdm(total=len(train_loader), desc="Training", disable=not accelerator.is_local_main_process)
        progress = tqdm(total=len(train_loader), desc="Training")
        model.train()
        for batch_idx, batch in enumerate(train_loader):  # tqdm(enumerate(sample1_loader), total=len(sample1_loader), desc="Epoch"):            
            optimizer.zero_grad(set_to_none=True)
            dv, text_embeddings, iv_s, iv_b, iv_w = batch
            image_embeddings, b, s, w = model(dv=dv, iv_s=iv_s, iv_b=iv_b, iv_w=iv_w)

            # Make entropic targets
            if eb is None:
                eb = F.softmax(torch.ones_like(b), 1)
                es = F.softmax(torch.ones_like(s), 1)
                ew = F.softmax(torch.ones_like(w), 1)
            loss = nn.CrossEntropyLoss()(image_embeddings, text_embeddings)
            bl = F.cross_entropy(b, eb)
            sl = F.cross_entropy(s, es)
            wl = F.cross_entropy(w, ew)
            loss = loss + bl + sl + wl

            # Optimize
            # accelerator.backward(loss)
            loss.backward()
            optimizer.step()
            scheduler.step()
            batch_losses.append(loss)
            progress.set_postfix({"train_loss": loss})  # , "compounds": comp_loss, "phenotypes": pheno_loss})
            progress.update()

        # Run test set
        test_losses, test_accs = [], []
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):  # tqdm(enumerate(sample1_loader), total=len(sample1_loader), desc="Epoch"):
                dv, text_embeddings, iv_s, iv_b, iv_w = batch
                image_embeddings, b, s, w = model(dv=dv, iv_s=iv_s, iv_b=iv_b, iv_w=iv_w)
                # image_embeddings = model(dv=dv, iv_s=iv_s, iv_b=iv_b, iv_w=iv_w)
                loss = nn.CrossEntropyLoss()(image_embeddings, text_embeddings)
                # Losses to become invariant to batch effects
                if teb is None:
                    teb = F.softmax(torch.ones_like(b), 1)
                    tes = F.softmax(torch.ones_like(s) / len(s), 1)
                    tew = F.softmax(torch.ones_like(w) / len(w), 1)
                bl = (b.argmax(1) == iv_b).float().mean()  # F.cross_entropy(b, teb)
                sl = (s.argmax(1) == iv_s).float().mean()  # F.cross_entropy(s, tes)
                wl = (w.argmax(1) == iv_w).float().mean()  # F.cross_entropy(w, tew)
                _, tk = torch.topk(image_embeddings, tops, dim=1)
                accuracy = (tk == text_embeddings[:, None]).sum(1).float().sum() / len(tk)
                test_losses.append(loss)
                test_accs.append(accuracy)

        # Check performances
        epoch_loss = np.mean([x.item() for x in batch_losses])
        test_loss = np.mean([x.item() for x in test_losses])
        test_acc = np.mean([x.item() for x in test_accs]) * 100.
        if 1:  # accelerator.is_main_process:
            if test_loss < best_loss:
                print("Saving best performing weights")
                best_loss = test_loss
                best_test_acc = test_acc
                torch.save(model.state_dict(), path)
                epoch_counter = 0
            else:
                epoch_counter += 1
            progress.set_postfix({"epoch": epoch, "number_compounds": nc, "train_loss": epoch_loss, "test_loss": test_loss, "test_acc": test_acc, "best_test_acc": best_test_acc, "well_loss": wl, "batch_loss": bl, "source_loss": sl})
            progress.update()
        progress.close()
        # accelerator.wait_for_everyone()
    print('Finished training')

    # Load best weights
    model.load_state_dict(torch.load(path))
    model.eval()

    # Run MoA test
    moa_perf = eval_tools.run(kind="moa", epochs=test_epochs)

    # Run Target test
    target_perf = eval_tools.run(kind="target", epochs=test_epochs)

    # Update the DB with results
    results = {
        "meta_id": id,
        "moa_acc": moa_perf["acc"],
        "moa_loss": moa_perf["loss"],
        "moa_acc_std": moa_perf["acc_std"],
        "moa_loss_std": moa_perf["loss_std"],
        "target_acc": target_perf["acc"],
        "target_loss": target_perf["loss"],
        "target_acc_std": target_perf["acc_std"],
        "target_loss_std": target_perf["loss_std"],
    }
    db_utils.record_performance(results)
    
    
if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Run in debug mode
        params = {
            "id": -1,
            "data_prop": np.arange(0, 1.1, 0.1)[-1],
            "label_prop": [0.1, 0.25, 0.5, 1.0][0],  # Proportion of labels, i.e. x% of molecules for labels
            "objective": ["mol_class", "masked_recon", "barlow"][0],
            "lr": [1e-3, 1e-4][0],
            "bs": [10000][0],
            "moa": [True][0],
            "target": [True][0],
            "layers": [1, 3, 6, 12, 24][0],
            "width": [64, 128, 512, 1024, 2048][0],
            "batch_effect_correct": [True, False][0],
        }
    else:
        # Run in DB mode
        params = db_utils.get_and_reserve_params()
    main(**params)

