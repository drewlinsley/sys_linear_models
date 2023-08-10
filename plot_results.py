import db_utils
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np


data = db_utils.get_all_results()
import pdb;pdb.set_trace()
df = pd.DataFrame.from_dict(data)
comb = ((df.moa_acc + df.target_acc) / 2).values
df["comb"] = comb
g1 = df.groupby([x for x in df.columns[:1]]).max("comb")
rs = [x for x in g1.columns]
rs.pop(rs.index("lr"))
g2 = g1.groupby(rs).max("comb").reset_index()
palette = sns.color_palette("flare_r", len(np.unique(df.data_prop)), as_cmap=True)
summ_df = df.groupby(['data_prop', 'label_prop', 'objective', 'layers', 'width', 'batch_effect_correct']).max().reset_index()
# sns.relplot(data=df, x="data_prop", y="comb", size="width", hue="label_prop", col="batch_effect_correct", kind="line", palette=palette, errorbar=None);plt.show()
sns.relplot(data=df, x="data_prop", y="comb", size="width", hue="label_prop", col="batch_effect_correct", kind="scatter", palette=palette);plt.show()

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

