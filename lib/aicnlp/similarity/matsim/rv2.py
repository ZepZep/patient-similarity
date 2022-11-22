from pathlib import Path
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from multiprocessing import Pool


def calculate(prefix, data_dir, input_file, patients, id_letter, workers=6):
    print(f"{prefix}loading", end="", flush=True)
    in_name = Path(input_file).with_suffix("").name
    records = pd.read_feather(input_file)

    if patients is not None:
        records = records[records["pid"].isin(set(patients))].copy()

    print(f", grouping", end="", flush=True)
    groups = records.groupby("pid")["vec"].apply(lambda x: np.vstack(x).T)

    print(f",", flush=True)
    sim = calculate_rv2(prefix, groups, workers)

    print(f"{prefix}writing {sim.shape[0]}x{sim.shape[1]} matrix", end="", flush=True)
    np.savez_compressed(
        f"{data_dir}/3/{id_letter}rv2-{in_name}.npz",
        sim=sim, patients=np.array(groups.index)
    )

    print(f", DDONE.", flush=True)


# adapted from the hoggorm python module
# https://hoggorm.readthedocs.io/en/latest/matrix_corr_coeff.html
def mat_worker(arr):
    scalArr = np.dot(arr, np.transpose(arr))
    diego = np.diag(np.diag(scalArr))
    scalArrMod = scalArr - diego
    return scalArrMod


# adapted from the hoggorm python module
# https://hoggorm.readthedocs.io/en/latest/matrix_corr_coeff.html
def sim_worker(args):
    i0, i1, mat0, mat1 = args
    nom = np.trace(
        np.dot(np.transpose(mat0),
                  mat1))
    denom1 = np.trace(
        np.dot(np.transpose(mat0),
                  mat0))
    denom2 = np.trace(
        np.dot(np.transpose(mat1),
                  mat1))
    Rv = nom / np.sqrt(denom1 * denom2)
    return i0, i1, Rv


# adapted from the hoggorm python module
# https://hoggorm.readthedocs.io/en/latest/matrix_corr_coeff.html
def calculate_rv2(prefix, dataList, workers):
    # First compute the scalar product matrices for each data set X
    scalArrList = []

    it = map(mat_worker, dataList)
    if prefix is not None:
        it  = tqdm(it, desc=f"{prefix}  Calculating scalArrMods", total=len(dataList),
                   bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}')
    for scalArrMod in it:
        scalArrList.append(scalArrMod)

    # Now compute the 'between study cosine matrix' C
    C = np.zeros((len(dataList), len(dataList)), np.float32)

    it = np.array(np.triu_indices(len(dataList), k=0)).T
    ita = ( (i0, i1, scalArrList[i0], scalArrList[i1]) for i0, i1 in it )
    if prefix is not None:
        ita = tqdm(ita, desc=f"{prefix}  Calculating  similarity", total=len(it),
                   bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}')

    with Pool(workers) as p:
        for i0, i1, rv in p.imap(sim_worker, ita, chunksize=20000):
            C[i0, i1] = rv
            C[i1, i0] = rv

    return C


def rv2(p1, p2):
    p1, p2 = p1.T, p2.T
    scalArrList = []
    for arr in (p1, p2):
        scalArr = np.dot(arr, np.transpose(arr))
        diego = np.diag(np.diag(scalArr))
        scalArrMod = scalArr - diego
        scalArrList.append(scalArrMod)

    nom = np.trace(
        np.dot(np.transpose(scalArrList[0]),
                    scalArrList[1]))
    denom1 = np.trace(
        np.dot(np.transpose(scalArrList[0]),
                    scalArrList[0]))
    denom2 = np.trace(
        np.dot(np.transpose(scalArrList[1]),
                    scalArrList[1]))
    Rv = nom / np.sqrt(denom1 * denom2)

    return Rv



def get_arrays(series):
    return series.groupby("pid").apply(lambda x: np.vstack(x).T)


def take_multi(df, sl, level=0):
    return df.loc[vectors.index.levels[level][sl].tolist()]
