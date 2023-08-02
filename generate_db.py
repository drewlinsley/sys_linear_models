"""
- training
+ data_prop [float]  # Measure scaling laws
+ objective [str]  # Molecule classification, masked-reconstruction, contrastive learning (b)
+ classification_proportion [float]  # Only for molecule classification task — change the number of unique molecules included
+ masking_proportion
+ learning rate

- evals (MLP-probe)
+ moa [bool]
+ target [bool]

- architecture
+ layers [int]
+ width [int]
+ batch_effect_correct [bool]
"""

import os
import psycopg
import itertools
import numpy as np
from tqdm import tqdm
from db_config import DBInfo
# https://docs.python.org/3/library/sqlite3.html


def get_all_combinations(exps):
    """Get all combinations of the values in the dictionary `exps`.

    Args:
    exps: The dictionary of experiments.

    Returns:
    A list of dictionaries, where each dictionary represents one combination of
    the values in `exps`.
    """

    combinations = []
    for values in itertools.product(*exps.values()):
        combination = dict(zip(exps.keys(), values))
        combinations.append(combination)

    return combinations


exps = {
    "data_prop": [0.5, 1.],  # [0.1, 0.2, 0.4, 0.6, 0.8, 1.],  # np.arange(0, 1.1, 0.1),
    "label_prop": [0.5, 1.],  # [0.1, 0.2, 0.4, 0.6, 0.8, 1.],  # np.arange(0, 1.1, 0.1),  # Proportion of labels, i.e. x% of molecules for labels
    "objective": ["mol_class", "masked_recon"],
    "lr": [1e-3, 1e-4],
    "bs": [5000],
    "moa": [True],
    "target": [True],
    "layers": [1, 3, 6],  # [1, 3, 6, 12],
    "width": [256, 512, 768],  # [256, 512, 768, 1024],
    "batch_effect_correct": [True, False],
}

combinations = get_all_combinations(exps)
print("Derived {} combinations.".format(len(combinations)))

# Create a DB
cfg = DBInfo()
db_name = cfg.db_name  # "chem_exps"
con = psycopg.connect(dbname=db_name, autocommit=True)
cur = con.cursor()
cur.execute("DROP TABLE IF EXISTS metadata")
cur.execute("CREATE TABLE metadata(id SERIAL PRIMARY KEY, reserved boolean, finished boolean, data_prop float, label_prop float, objective varchar, lr float, bs int, moa boolean, target boolean, layers int, width int, batch_effect_correct boolean)")
cur.execute("DROP TABLE IF EXISTS results")
cur.execute("CREATE TABLE results(id SERIAL PRIMARY KEY, meta_id int, moa_acc float, moa_loss float, moa_acc_std float, moa_loss_std float, target_acc float, target_loss float, target_acc_std float, target_loss_std float)")
for idx, combo in tqdm(enumerate(combinations), total=len(combinations), desc="Preparing inserts"):
    cols = [x for x in combo.keys()]
    vals = [x for x in combo.values()]
    cols = ["reserved", "finished"] + cols
    # vals = [idx, False, False] + vals
    vals = [False, False] + vals
    # cols = tuple(cols)
    # vals = tuple(vals)
    query = """INSERT INTO metadata(%s) VALUES(%%s, %%s, %%s, %%s, %%s, %%s, %%s, %%s, %%s, %%s, %%s, %%s); """ % (', '.join(cols))
    # cur.executemany(query, combo)
    cur.execute(query, vals)
con.commit()

# res = cur.execute("SELECT * from metadata")
# print(res.fetchall())
con.close()
print("Successfully created DB.")

