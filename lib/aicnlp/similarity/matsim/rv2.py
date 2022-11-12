from pathlib import Path
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from multiprocessing import Pool


def calculate(prefix, data_dir, input_file, patients, id_letter):
    print(f"{prefix}loading", end="", flush=True)
    in_name = Path(input_file).with_suffix("").name
    records = pd.read_feather(input_file)

    if patients is not None:
        records = records[records["pid"].isin(set(patients))].copy()

    print(f", grouping", end="", flush=True)
    groups = records.groupby("pid")["vec"].apply(lambda x: np.vstack(x).T)

    print(f",", flush=True)
    sim = calculate_rv2(prefix, groups)

    print(f"{prefix}writing {sim.shape[0]}x{sim.shape[1]} matrix", end="", flush=True)
    np.savez_compressed(
        f"{data_dir}/3/{id_letter}rv2-{in_name}.npz",
        sim=sim, patients=np.array(groups.index)
    )

    print(f", DDONE.", flush=True)


def mat_worker(arr):
    scalArr = np.dot(arr, np.transpose(arr))
    diego = np.diag(np.diag(scalArr))
    scalArrMod = scalArr - diego
    return scalArrMod


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


def calculate_rv2(prefix, dataList):
    """
    This function computes the RV matrix correlation coefficients between pairs
    of arrays. The number and order of objects (rows) for the two arrays must
    match. The number of variables in each array may vary. The RV2 coefficient
    is a modified version of the RV coefficient with values -1 <= RV2 <= 1.
    RV2 is independent of object and variable size.
    Reference: `Matrix correlations for high-dimensional data - the modified RV-coefficient`_
    .. _Matrix correlations for high-dimensional data - the modified RV-coefficient: https://academic.oup.com/bioinformatics/article/25/3/401/244239
    PARAMETERS
    ----------
    dataList : list
        A list holding an arbitrary number of numpy arrays for which the RV
        coefficient will be computed.
    RETURNS
    -------
    numpy array
        A list holding an arbitrary number of numpy arrays for which the RV
        coefficient will be computed.
    """

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

    it = np.array(np.triu_indices(len(dataList), k=1)).T
    ita = ( (i0, i1, scalArrList[i0], scalArrList[i1]) for i0, i1 in it )
    itp = map(sim_worker, ita)
    if prefix is not None:
        itp = tqdm(itp, desc=f"{prefix}  Calculating  similarity", total=len(it),
                   bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}')
    for i0, i1, rv in itp:
        C[i0, i1] = rv
        C[i1, i0] = rv

    return C


def get_arrays(series):
    return series.groupby("pid").apply(lambda x: np.vstack(x).T)

def take_multi(df, sl, level=0):
    return df.loc[vectors.index.levels[level][sl].tolist()]

