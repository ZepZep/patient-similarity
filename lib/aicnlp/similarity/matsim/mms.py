from pathlib import Path
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from multiprocessing import Pool
from sklearn.metrics.pairwise import cosine_similarity


def calculate(prefix, data_dir, input_file, patients, id_letter):
    print(f"{prefix}loading", end="", flush=True)
    in_name = Path(input_file).with_suffix("").name
    records = pd.read_feather(input_file)

    if patients is not None:
        records = records[records["pid"].isin(set(patients))].copy()

    print(f", grouping", end="", flush=True)
    groups = records.groupby("pid")["vec"].apply(lambda x: np.vstack(x))
    groups_list = groups.to_list()

    print(f",", flush=True)
    sim = np.zeros((len(groups), len(groups)), np.float32)
    it = np.array(np.triu_indices(len(groups), k=0)).T
    for i0, i1 in tqdm(it, desc=f"{prefix}  Calculating  similarity"):
        if i0 == i1:
            sim[i0, i1] = 1
            continue

        cur_sim =  mms(groups_list[i0], groups_list[i1])
        sim[i0, i1] = cur_sim
        sim[i1, i0] = cur_sim

    print(f"{prefix}writing {sim.shape[0]}x{sim.shape[1]} matrix", end="", flush=True)
    np.savez_compressed(
        f"{data_dir}/3/{id_letter}mms-{in_name}.npz",
        sim=sim, patients=np.array(groups.index)
    )

    print(f", DDONE.", flush=True)

def mms(p1, p2):
    cur_sim_mat = cosine_similarity(p1, p2)
    maxes = np.hstack([
        cur_sim_mat.max(axis=0),
        cur_sim_mat.max(axis=1),
    ])
    return maxes.mean()
