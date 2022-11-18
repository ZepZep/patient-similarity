from pathlib import Path
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from multiprocessing import Pool
from sklearn.metrics.pairwise import cosine_similarity


def calculate(prefix, data_dir, input_file, patients, id_letter, workers=12):
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
    itlen = len(it)
    it = ((i0, i1, groups_list[i0], groups_list[i1]) for i0, i1 in it)
    it = tqdm(it, desc=f"{prefix}  Calculating  similarity", total=itlen)
    with Pool(workers) as p:
        for i0, i1, cur_sim in p.imap(worker_eds, it):
            sim[i0, i1] = cur_sim
            sim[i1, i0] = cur_sim

    print(f"{prefix}writing {sim.shape[0]}x{sim.shape[1]} matrix", end="", flush=True)
    np.savez_compressed(
        f"{data_dir}/3/{id_letter}eds-{in_name}.npz",
        sim=sim, patients=np.array(groups.index)
    )

    print(f", DDONE.", flush=True)


def worker_eds(args):
    i0, i1, p0, p1 = args
    if i0 == i1:
        return i0, i1, 1
    return i0, i1, eds(p0, p1)

def mms(p1, p2):
    cur_sim_mat = cosine_similarity(p1, p2)
    maxes = np.hstack([
        cur_sim_mat.max(axis=0),
        cur_sim_mat.max(axis=1),
    ])
    return maxes.mean()


def eds(p1, p2, return_path=False):
    sims = cosine_similarity(p1, p2)
    sims_plus = (sims + 1 ) / 2

    scores = np.zeros(sims_plus.shape, dtype=float)
    path = {}
    scores[0,0] = sims_plus[0,0]
    for r in range(1, scores.shape[0]):
        scores[r,0] = scores[r-1,0] * sims_plus[r,0]
        path[(r,0)] = (r-1,0)
    for c in range(1, scores.shape[1]):
        scores[0,c] = scores[0,c-1] * sims_plus[0,c]
        path[(0,c)] = (0,c-1)


    for r in range(1, scores.shape[0]):
        for c in range(1, scores.shape[1]):
            options = get_options(scores, r, c)
            best_score, pr, pc = max(options)
            scores[r,c] = sims_plus[r,c]*best_score
            path[(r,c)] = (pr, pc)

    final_pos = scores.shape[0]-1, scores.shape[1]-1
    best_path = reconstruct_path(path, final_pos)
    # final_score = scores[-1, -1]
    # similarity_score = final_score**(1/len(best_path))

    similarity_score = sum(sims[r,c] for r, c in best_path) / len(best_path)

    if return_path:
        return similarity_score, best_path
    return similarity_score


deltas = [(-1,-1), (-1,0), (0,-1)]
def get_options(scores, r, c):
    return [
        (scores[r+dr, c+dc], r+dr, c+dc)
        for dr, dc in deltas
    ]

def reconstruct_path(path, start):
    pos = start
    rp = [pos]
    while True:
        nextpos = path.get(pos, None)
        if nextpos is None:
            if pos != (0,0):
                print(f"Could not reconstruct path, ended at {pos}")
            return rp
        pos = nextpos
        rp.append(pos)
