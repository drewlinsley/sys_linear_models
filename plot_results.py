import os
import db_utils
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from decimal import Decimal
import matplotlib as mpl
from scipy.stats import linregress
import statsmodels.formula.api as sm


def get_aux_data(idf, counts, dcounts):
    add_params, total_data = [], []
    for idx in range(len(idf)):
        r = idf.iloc[idx]
        w = r["width"]
        l = r["layers"]
        d = r["data_prop"]
        b = r["label_prop"]
        params = counts[np.logical_and(counts.width == w, counts.layers == l)].counts.values
        datas = dcounts[np.logical_and(dcounts.data_prop == d, dcounts.label_prop == b)].total_data.values
        add_params.append(params)
        total_data.append(datas)
    add_params = np.concatenate(add_params)
    total_data = np.concatenate(total_data)
    idf["params"] = add_params
    idf["total_data"] = total_data
    return idf


def func(x, a, b):
    return (a / x) ** -b
from scipy.optimize import curve_fit


def fig_scale_data_labels(dv, ylim, text=True, agg="max", powerlaw=False, data="all"):
    f, axs = plt.subplots(1, 1)
    if data.lower() == "cp":
        df = cp_df
    if data.lower() != "cp":
        plt.axhline(cp_data[dv].values, color="black", linestyle='--', lw=1)
        plt.axhline(cp_df[dv].max(), color="black", linestyle='--', lw=1)
        rect = plt.Rectangle((0, cp_data[dv].values.tolist()[0]), 800000, (cp_df[dv].max() - cp_data[dv].values)[0], color='black', alpha=0.05)
        axs.add_patch(rect)
    if text:
        if data.lower() != "cp":
            plt.text(50000, cp_data[dv].values + 1, "Recursion (Cellprofiler)")
        sns.lineplot(data=df, x="total_data", y=dv, hue="label_prop", palette=palette, ax=axs, estimator=agg)
    else:
        sns.lineplot(data=df, x="total_data", y=dv, hue="label_prop", palette=palette, ax=axs, estimator=agg, legend=False)

    if agg != "max":
        v = df.groupby(["label_prop"]).agg(agg).reset_index()[dv]
        rd = df.groupby(["label_prop"]).agg("max").reset_index()
        rd[dv] = v
    else:
        rd = df.groupby(["label_prop"]).agg(agg).reset_index()
    X = np.stack((rd.total_data.values, rd.label_prop.values), 1)
    rd["adj_labels"]=(rd.total_data - 118949)
    if powerlaw:
        # fit = np.polyfit(rd["adj_labels"], rd[dv], 1)
        # slope, intercept = fit
        popt, pcov = curve_fit(func, rd["adj_labels"], rd[dv], method="dogbox")
    else:
        m = sm.ols(formula="{} ~ adj_labels".format(dv), data=rd).fit()
        intercept = m.params.Intercept
        slope = m.params.adj_labels

    # slope, intercept, r_value, p_value, std_err = linregress(rd.label_prop, rd.target_acc)
    # y_predicted = slope * rd.total_data + intercept
    if powerlaw:
        first_predicted = func(rd["adj_labels"].min(), popt[0], popt[1])
        final_predicted = func(rd["adj_labels"].max(), popt[0], popt[1]) 
        plt.plot([0, rd.adj_labels.max() + 800000], [first_predicted, final_predicted], color="#c90657", linestyle="-", lw=1, alpha=0.5)
    else:
        final_predicted = slope * (rd.adj_labels.max() + 800000) + intercept
        plt.plot([0, rd.adj_labels.max() + 800000], [intercept, final_predicted], color="#c90657", linestyle="-", lw=1, alpha=0.5)
    # rd.total_data, y_predicted, "k--")

    chance = 100 * (1/1282.)
    # plt.axhline(chance, color="black", linestyle='--', lw=1, alpha=0.5)
    # plt.text(50000, chance + 1, "Chance")
    # plt.annotate("$Accuracy = {:.2e} * (# of Novel Molecules) + {:.2f}$".format(Decimal(slope)), xy=(50000, 75))
    print("$Accuracy = {:.2e} * (# of Novel Molecules) + {:.2f}$".format(Decimal(slope)))
    plt.xlim([0, 800000])
    plt.ylim(ylim)
    if text:
        plt.legend(loc='center right')
    plt.ylabel("{} deconvolution accuracy".format(dv))
    plt.xlabel("Total number of training phenotypes")
    plt.xticks(rotation = -30)
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    if data.lower() != "cp":
        plt.savefig(os.path.join(results_dir, "fig_{}_scale_data_labels_text_{}.png".format(dv, text)), dpi=300)
    else:
        plt.savefig(os.path.join(results_dir, "cp_fig_{}_scale_data_labels_text_{}.png".format(dv, text)), dpi=300)
    plt.show()
    plt.close(f)


def fig_scale_data_fits(dv, ylim, text=True, agg="max", data="all"):
    f, axs = plt.subplots(1, 1)
    if data.lower() == "cp":
        df = cp_df
    if text:
        if data.lower() != "cp":
            plt.text(50000, cp_data[dv].values + 1, "Recursion (Cellprofiler)")
        sns.lineplot(data=df, x="total_data", y=dv, hue="label_prop", palette=palette, ax=axs, estimator=agg, alpha=0.2)
    else:
        sns.lineplot(data=df, x="total_data", y=dv, hue="label_prop", palette=palette, ax=axs, estimator=agg, legend=False, alpha=0.2)

    if agg != "max":
        v = df.groupby(["label_prop", "data_prop"]).agg(agg).reset_index()[dv]
        rd = df.groupby(["label_prop", "data_prop"]).agg("max").reset_index()
        rd[dv] = v
    else:
        rd = df.groupby(["label_prop", "data_prop"]).agg(agg).reset_index()
    X = np.stack((rd.total_data.values, rd.label_prop.values), 1)
    # rd["adj_labels"]=(rd.total_data - 118949)
    rd["adj_labels"] = rd.total_data

    # Seperate fits for each data_prop
    fits = {}
    for d in np.sort(rd.data_prop.unique()):
        m = sm.ols(formula="{} ~ adj_labels".format(dv), data=rd[rd.data_prop == d]).fit()
        intercept = m.params.Intercept
        slope = m.params.adj_labels
        fits[d] = [intercept, slope]
        print("Accuracy = {:.2e} * (# of Novel Molecules) + {:.2f}".format(Decimal(slope), intercept))
        final_predicted = slope * (rd[rd.data_prop == d].adj_labels.max() + 800000) + intercept
        plt.plot([0, rd[rd.data_prop == d].adj_labels.max() + 800000], [intercept, final_predicted], color="#c90657", linestyle="-", lw=1, alpha=0.5)

    # chance = 100 * (1/1282.)
    # plt.axhline(chance, color="black", linestyle='--', lw=1, alpha=0.5)
    # plt.text(50000, chance + 1, "Chance")
    # plt.annotate("$Accuracy = {:.2e} * (# of Novel Molecules) + {:.2f}$".format(Decimal(slope)), xy=(50000, 75))
    plt.xlim([0, 800000])
    plt.ylim(ylim)
    if text:
        plt.legend(loc='center right')
    plt.ylabel("{} deconvolution accuracy".format(dv))
    plt.xlabel("Total number of training phenotypes")
    plt.xticks(rotation = -30)
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    if data.lower() == "cp":
        plt.savefig(os.path.join(results_dir, "fig_{}_multi_data_fits_text_{}.png".format(dv, text)), dpi=300)
    else:
        plt.savefig(os.path.join(results_dir, "fig_{}_multi_data_fits_text_{}.png".format(dv, text)), dpi=300)
    plt.show()
    plt.close(f)


def fig_scale_params_depth(dv, ylim, text=True, palette="crest", agg="max", data="all"):
    f, axs = plt.subplots(1, 1)
    if data.lower() == "cp":
        df = cp_df
    if text:
        if data.lower() != "cp":
            plt.text(df.params.min() + 50000, cp_data[dv].values + 1, "Recursion (Cellprofiler)")
        sns.lineplot(data=df, x="params", y=dv, hue="layers", palette="viridis", estimator=agg)
    else:
        sns.lineplot(data=df, x="params", y=dv, hue="layers", palette="viridis", estimator=agg, legend=False)
        if data.lower() != "cp":
            plt.axhline(cp_data[dv].values, color="black", linestyle='--', lw=1)
            plt.axhline(cp_df[dv].max(), color="black", linestyle='--', lw=1)
            rect = plt.Rectangle((df.params.min(), cp_data[dv].values.tolist()[0]), (df.params.max() - df.params.min()), (cp_df[dv].max() - cp_data[dv].values)[0], color='black', alpha=0.05)
            axs.add_patch(rect)

    if agg != "max":
        v = df.groupby(["params", "layers"]).agg(agg).reset_index()[dv]
        rd = df.groupby(["params", "layers"]).agg("max").reset_index()
        rd[dv] = v
    else:
        rd = df.groupby(["params", "layers"]).agg(agg).reset_index()
    m = sm.ols(formula="{} ~ params".format(dv), data=rd).fit()
    intercept = m.params.Intercept
    slope = m.params.params

    print("Accuracy = {:.2e} * (# of Novel Molecules) + {:.2f}".format(Decimal(slope), intercept))
    final_predicted = slope * (rd.params.max()) + intercept
    plt.plot([rd.params.min(), rd.params.max()], [intercept, final_predicted], color="#c90657", linestyle="-", lw=1, alpha=0.5)

    chance = 100 * (1/1282.)
    # if text:
    #     plt.text(50000, 75, "Accuracy = {:.2e} * (# of Novel Molecules) + {:.2f}".format(Decimal(slope), intercept))
    # plt.annotate("$Accuracy = {:.2e} * (# of Novel Molecules) + {:.2f}$".format(Decimal(slope)), xy=(50000, 75))
    plt.xlim([df.params.min(), df.params.max()])
    plt.ylim(ylim)
    plt.ylabel("{} deconvolution accuracy".format(dv))
    plt.xlabel("Number of parameters")
    plt.xticks(rotation = -30)
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    plt.savefig(os.path.join(results_dir, "fig_{}_scale_params_depth_text_{}.png".format(dv, text)), dpi=300)
    plt.show()
    plt.close(f)


def fig_scale_data_data_fits(dv, ylim, text=True, agg="max", data="all"):
    f, axs = plt.subplots(1, 1)
    if data.lower() == "cp":
        df = cp_df
    if text:
        if data.lower() != "cp":
            plt.text(50000, cp_data[dv].values + 1, "Recursion (Cellprofiler)")
        sns.lineplot(data=df, x="total_data", y=dv, hue="data_prop", palette=palette, ax=axs, estimator=agg, alpha=0.2)
    else:
        sns.lineplot(data=df, x="total_data", y=dv, hue="data_prop", palette=palette, ax=axs, estimator=agg, legend=False, alpha=0.2)

    if agg != "max":
        v = df.groupby(["label_prop", "data_prop"]).agg(agg).reset_index()[dv]
        rd = df.groupby(["label_prop", "data_prop"]).agg("max").reset_index()
        rd[dv] = v
    else:
        rd = df.groupby(["label_prop", "data_prop"]).agg(agg).reset_index()
    X = np.stack((rd.total_data.values, rd.label_prop.values), 1)
    # rd["adj_labels"]=(rd.total_data - 118949)
    rd["adj_labels"] = rd.total_data

    # Seperate fits for each data_prop
    fits = {}
    for d in np.sort(rd.data_prop.unique()):
        m = sm.ols(formula="{} ~ adj_labels".format(dv), data=rd[rd.data_prop == d]).fit()
        intercept = m.params.Intercept
        slope = m.params.adj_labels
        fits[d] = [intercept, slope]
        print("Accuracy = {:.2e} * (# of Novel Molecules) + {:.2f}".format(Decimal(slope), intercept))
        final_predicted = slope * (rd[rd.data_prop == d].adj_labels.max() + 800000) + intercept
        plt.plot([0, rd[rd.data_prop == d].adj_labels.max() + 800000], [intercept, final_predicted], color="#c90657", linestyle="-", lw=1, alpha=0.5)

    # chance = 100 * (1/1282.)
    # plt.axhline(chance, color="black", linestyle='--', lw=1, alpha=0.5)
    # plt.text(50000, chance + 1, "Chance")
    # plt.annotate("$Accuracy = {:.2e} * (# of Novel Molecules) + {:.2f}$".format(Decimal(slope)), xy=(50000, 75))
    plt.xlim([0, 800000])
    plt.ylim(ylim)
    if text:
        plt.legend(loc='center right')
    plt.ylabel("{} deconvolution accuracy".format(dv))
    plt.xlabel("Total number of training phenotypes")
    plt.xticks(rotation = -30)
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    if data.lower() == "cp":
        plt.savefig(os.path.join(results_dir, "fig_{}_multi_data_fits_text_{}.png".format(dv, text)), dpi=300)
    else:
        plt.savefig(os.path.join(results_dir, "fig_{}_multi_data_fits_text_{}.png".format(dv, text)), dpi=300)
    plt.show()
    plt.close(f)


results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
data = db_utils.get_all_results()
cp_data = db_utils.get_all_results(cp=True)
df = pd.DataFrame.from_dict(data)
cp_df = pd.DataFrame.from_dict(cp_data)
counts = pd.read_csv("param_counts.csv")
dcounts = pd.read_csv("data_counts.csv")
comb = ((df.moa_acc + df.target_acc) / 2).values
df["comb"] = comb
df.groupby("label_prop")

df = get_aux_data(df, counts, dcounts)
cp_df = get_aux_data(cp_df, counts, dcounts)

g1 = df.groupby([x for x in df.columns[:1]]).max("comb")
rs = [x for x in g1.columns]
rs.pop(rs.index("lr"))
g2 = g1.groupby(rs).max("comb").reset_index()
palette = sns.color_palette("flare_r", len(np.unique(df.label_prop)), as_cmap=True)
palette = "flare_r"  # None
cp_palette = "crest_r"
summ_df = df.groupby(['data_prop', 'label_prop', 'objective', 'layers', 'width', 'batch_effect_correct']).max().reset_index()
# sns.relplot(data=df, x="data_prop", y="comb", size="width", hue="label_prop", col="batch_effect_correct", kind="line", palette=palette, errorbar=None);plt.show()
moa_min, moa_max = df.moa_acc.min() - 0.1, df.moa_acc.max() + 0.1
target_min, target_max = df.target_acc.min() - 0.1, df.target_acc.max() + 0.1
re_min, re_max = df.rediscovery_acc.min() - 0.05, df.rediscovery_acc.max() + 0.05
orf_min, orf_max = df.d_orf.min() - 0.02, df.d_orf.max() + 0.02
crispr_min, crispr_max = df.d_crispr.min() - 0.02, df.d_crispr.max() + 0.02
task_min, task_max = 5, df.task_loss.max() + 1
df.sort_values("task_loss")

cp_data = df[df.cell_profiler]
df = df[~df.cell_profiler]
df.target_acc *= 100
cp_data.target_acc *= 100
cp_df.target_acc *= 100
df.moa_acc *= 100
cp_data.moa_acc *= 100
cp_df.moa_acc *= 100

import matplotlib.cm as cm

# fig_scale_data_labels(dv="target_acc")
# fig_scale_data_labels(dv="target_acc", ylim=[40, 70], text=False)
# fig_scale_params_depth(dv="target_acc", ylim=[40, 70])
# fig_scale_params_depth(dv="target_acc", ylim=[40, 70], text=False)

# fig_scale_data_labels(dv="moa_acc", ylim=[20, 55])
# fig_scale_data_labels(dv="moa_acc", ylim=[20, 55], text=False)
# fig_scale_params_depth(dv="moa_acc", ylim=[20, 55], text=False)

# fig_scale_data_labels(dv="target_loss", ylim=[3.5, 7.3], agg="min")
# fig_scale_data_labels(dv="target_loss", ylim=[3.5, 7.3], text=False, agg="min")
# fig_scale_params_depth(dv="target_loss", ylim=[3.5, 7.3], agg="min")
# fig_scale_params_depth(dv="target_loss", ylim=[3.5, 7.3], text=False, agg="min")

# fig_scale_data_labels(dv="rediscovery_acc", ylim=[3.5, 7.3], agg="min")
# fig_scale_data_labels(dv="rediscovery_z", ylim=[3.5, 7.3], agg="min")
# fig_scale_data_labels(dv="z_prime_crispr", ylim=[3.5, 7.3], agg="min")
# fig_scale_data_labels(dv="z_prime_orf", ylim=[3.5, 7.3], agg="min")

# fig_scale_data_labels(dv="target_loss", ylim=[3.5, 7.3], text=False, agg="min")
# fig_scale_data_labels(dv="target_loss", ylim=[3.5, 7.3], text=True, agg="min", powerlaw=False)
# fig_scale_params_depth(dv="target_loss", ylim=[3.5, 7.3], agg="min")
# fig_scale_params_depth(dv="target_loss", ylim=[3.5, 7.3], text=False, agg="min")

# fig_scale_data_fits(dv="moa_acc", ylim=[20, 55], agg="max")
# fig_scale_data_fits(dv="moa_acc", ylim=[20, 55], agg="max", text=False)

# CP figs
# fig_scale_data_fits(dv="moa_acc", ylim=[20, 55], agg="max", data="cp")
# fig_scale_data_fits(dv="moa_acc", ylim=[20, 55], agg="max", text=False, data="cp")
# fig_scale_params_depth(dv="moa_acc", ylim=[20, 55], text=False, data="cp")
# fig_scale_data_labels(dv="moa_acc", ylim=[20, 55], data="cp")
# fig_scale_data_labels(dv="moa_acc", ylim=[20, 55], data="cp", text=False)

fig_scale_data_data_fits(dv="moa_acc", ylim=[20, 55], agg="max")
plt.subplot(121)
dv = "moa_acc"
sns.lineplot(data=df, x="total_data", y=dv, hue="label_prop", palette=palette)
plt.subplot(122)
sns.lineplot(data=df, x="total_data", y=dv, hue="data_prop", palette=palette)
plt.show()
os._exit(1)


# Data vs target
sns.lmplot(data=df, x="total_data", y="target_acc", hue="label_prop", palette=palette, logx=True);plt.show()
sns.lineplot(data=df, x="params", y="target_acc", hue="label_prop", size="layers", palette=palette, estimator="max");plt.show()
sns.lineplot(data=df, x="params", y="target_acc", hue="data_prop", size="layers", palette=palette, estimator="max");plt.show()


# MOA perf vs. model size
f = plt.figure()
ax = plt.subplot(1, 1, 1)
cmap = cm.get_cmap(palette, len(np.unique(df.data_prop)))
for i, l in enumerate(np.unique(df.data_prop)):
    ddf = df[df.data_prop == l]
    sns.regplot(ddf, x="params", y="moa_acc", color=cmap(i), ax=ax)

for idx, d in enumerate(np.unique(df.label_prop)):
    ax = plt.subplot(1, 5, idx + 1)
    # sns.scatterplot(data=df[df.data_prop==d], x="params", y="moa_acc", hue="label_prop", palette=palette)
    if idx > 0:
        idf = df[df.label_prop == d]
        sns.scatterplot(data=idf, x="params", y="moa_acc", hue="data_prop", palette=palette, legend=False, ax=ax)
        cmap = cm.get_cmap(palette, len(np.unique(idf.data_prop)))
        for i, l in enumerate(np.unique(idf.data_prop)):
            ddf = idf[idf.data_prop == l]
            sns.regplot(ddf, x="params", y="moa_acc", color=cmap(i), ax=ax)
        # sns.lmplot(data=df[df.label_prop==d], x="params", y="moa_acc", hue="data_prop", palette=palette, legend=False)  # , ax=ax)
        # sns.regplot(data=df[df.label_prop==d], x="params", y="moa_acc", hue="data_prop", palette=palette, legend=False, ax=ax)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_yticklabels([])
        plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
        ax.set_ylabel("")
    else:
        idf = df[df.label_prop == d]
        sns.scatterplot(data=idf, x="params", y="moa_acc", hue="data_prop", palette=palette, legend=True, ax=ax)
        cmap = cm.get_cmap(palette, len(np.unique(idf.data_prop)))
        for i, l in enumerate(np.unique(idf.data_prop)):
            ddf = idf[idf.data_prop == l]
            sns.regplot(ddf, x="params", y="moa_acc", color=cmap(i), ax=ax)
        # sns.lmplot(data=df[df.label_prop==d], x="params", y="moa_acc", hue="data_prop", palette=palette, legend=True)  # , ax=ax)
        # sns.lineplot(data=df[df.label_prop==d], x="params", y="moa_acc", hue="data_prop", palette=palette, legend=False, ax=ax)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.title("Label proportion: {}".format(d))
    plt.ylim([0.1, 0.6])
plt.show()


from scipy.spatial import ConvexHull

hull = ConvexHull(points)


# MOA perf vs. model size
for idx, d in enumerate(np.unique(df.data_prop)):
    plt.subplot(1, 5, idx + 1)
    # sns.scatterplot(data=df[df.data_prop==d], x="params", y="moa_acc", hue="label_prop", palette=palette)
    if idx > 0:
        sns.scatterplot(data=df[df.data_prop==d], x="params", y="moa_acc", hue="label_prop", palette=palette, legend=False)
    else:
        sns.scatterplot(data=df[df.data_prop==d], x="params", y="moa_acc", hue="label_prop", palette=palette, legend=True)
    plt.title("Data proportion: {}".format(d))
    plt.ylim([0.1, 0.6])
plt.show()


# Measure performance wrt width/layers
datas = np.sort(df.data_prop.unique())
# df["params"] = df.width * df.data_prop
f, ax = plt.subplots(1, 1)
for d in datas:
    idf = df[df.data_prop == d]
    idf["data_prop"] = idf["data_prop"] + 1
    sns.lineplot(data=idf, x="data_prop", y="moa_acc", hue="label_prop", ax=ax)
plt.show()
plt.close(f)

f, ax = plt.subplots(1, 1)
for d in datas:
    idf = df[df.data_prop == d]
    idf["data_prop"] = idf["data_prop"] + 1
    sns.lineplot(data=idf, x="data_prop", y="moa_acc", hue="params", ax=ax)
plt.show()
plt.close(f)


# df["params"] = df.layers * df.width * df.data_prop
# sns.lineplot(data=df, x="params", y="moa_acc", hue="label_prop", size="width", palette=palette);plt.axhline(cp_data.moa_acc.values, color="black", linestyle='--', lw=1);plt.text(0.8, cp_data.moa_acc.values + 0.01, "Cellprofiler");plt.show()




f = plt.figure()
ud = np.sort(df.data_prop.unique())
pareto = []
dv = "moa_acc"
for idx, d in enumerate(ud):
    ax = plt.subplot(6, len(ud), idx + 1)
    ax.set_title("Data prop: {}".format(d))

    idf = df[np.logical_and(df.data_prop == d, df.batch_effect_correct == True)]
    pareto.append(idf[dv].max())
    dv = "moa_acc"
    sns.lineplot(data=idf, x="layers", y=dv, hue="label_prop", size="width", palette=palette, alpha=0.8, ax=ax, legend=False)
    ax.set_ylim([moa_min, moa_max])

    ax = plt.subplot(6, len(ud), idx + 1 + len(ud) * 1)
    dv = "target_acc"
    sns.lineplot(data=idf, x="layers", y=dv, hue="label_prop", size="width", palette=palette, alpha=0.8, ax=ax, legend=False)
    ax.set_ylim([target_min, target_max])

    ax = plt.subplot(6, len(ud), idx + 1 + len(ud) * 2)
    dv = "rediscovery_acc"
    sns.lineplot(data=idf, x="layers", y=dv, hue="label_prop", size="width", palette=palette, alpha=0.8, ax=ax, legend=False)
    ax.set_ylim([re_min, re_max])

    ax = plt.subplot(6, len(ud), idx + 1 + len(ud) * 3)
    dv = "d_orf"
    sns.lineplot(data=idf, x="layers", y=dv, hue="label_prop", size="width", palette=palette, alpha=0.8, ax=ax, legend=False)
    ax.set_ylim([orf_min, orf_max])

    ax = plt.subplot(6, len(ud), idx + 1 + len(ud) * 4)
    dv = "d_crispr"
    sns.lineplot(data=idf, x="layers", y=dv, hue="label_prop", size="width", palette=palette, alpha=0.8, ax=ax, legend=False)
    ax.set_ylim([crispr_min, crispr_max])

    ax = plt.subplot(6, len(ud), idx + 1 + len(ud) * 5)
    dv = "task_loss"
    sns.lineplot(data=idf, x="layers", y=dv, hue="label_prop", size="width", palette=palette, alpha=0.8, ax=ax, legend=False)
    ax.set_ylim([task_min, task_max])



# Add pareto plot here somehow
# On second thought, put all the layers on the x axis. Then relabel. That way we can draw as one big plot with a single pareto front
plt.show()

sns.relplot(data=df[df.data_prop == 1], x="layers", y="moa_acc", hue="label_prop", size="width", col="batch_effect_correct", kind="line", palette=palette);plt.show()
sns.relplot(data=df[df.data_prop == 1], x="layers", y="target_acc", hue="label_prop", size="width", col="batch_effect_correct", kind="line", palette=palette);plt.show()
sns.relplot(data=df[df.data_prop == 1], x="layers", y="rediscovery_acc", hue="label_prop", size="width", col="batch_effect_correct", kind="line", palette=palette);plt.show()
sns.relplot(data=df[df.data_prop == 1], x="layers", y="rediscovery_z", hue="label_prop", size="width", col="batch_effect_correct", kind="line", palette=palette);plt.show()
sns.relplot(data=df[df.data_prop == 1], x="layers", y="z_prime_orf", hue="label_prop", size="width", col="batch_effect_correct", kind="line", palette=palette);plt.show()
sns.relplot(data=df[df.data_prop == 1], x="layers", y="z_prime_crispr", hue="label_prop", size="width", col="batch_effect_correct", kind="line", palette=palette);plt.show()
sns.relplot(data=df[df.data_prop == 1], x="layers", y="d_orf", hue="label_prop", size="width", col="batch_effect_correct", kind="line", palette=palette);plt.show()
sns.relplot(data=df[df.data_prop == 1], x="layers", y="d_crispr", hue="label_prop", size="width", col="batch_effect_correct", kind="line", palette=palette);plt.show()


# Index(['data_prop', 'label_prop', 'objective', 'layers', 'width',
#        'batch_effect_correct', 'id', 'reserved', 'finished', 'lr', 'bs', 'moa',
#        'target', 'meta_id', 'moa_acc', 'moa_loss', 'moa_acc_std',
#        'moa_loss_std', 'target_acc', 'target_loss', 'target_acc_std',
#        'target_loss_std'],






f = plt.figure()
plt.subplot(3, 3, 1)
plt.scatter(summ_df.data_prop, summ_df.moa_acc, c=summ_df.label_prop)
plt.title("Data prop vs. Moa")

plt.subplot(3, 3, 2)
plt.scatter(summ_df.data_prop, summ_df.target_acc, c=summ_df.label_prop)
plt.title("Data prop vs. Target")

plt.subplot(3, 3, 3)
plt.scatter(summ_df.label_prop, summ_df.moa_acc, c=summ_df.label_prop)
plt.title("Label prop vs. Moa")

plt.subplot(3, 3, 4)
plt.scatter(summ_df.label_prop, summ_df.target_acc, c=summ_df.label_prop)
plt.title("Label prop vs. Target")

plt.subplot(3, 3, 5)
plt.scatter(summ_df.layers, summ_df.moa_acc, c=summ_df.data_prop)
plt.title("Layers vs. Moa")

plt.subplot(3, 3, 6)
plt.scatter(summ_df.layers, summ_df.target_acc, c=summ_df.data_prop)
plt.title("Layers vs. Target")

plt.subplot(3, 3, 7)
plt.scatter(summ_df.width, summ_df.moa_acc, c=summ_df.data_prop)
plt.title("Width vs. Moa")

plt.subplot(3, 3, 8)
plt.scatter(summ_df.width, summ_df.target_acc, c=summ_df.data_prop)
plt.title("Width vs. Target")

plt.show()

