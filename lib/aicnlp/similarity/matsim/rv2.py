import pandas as pd
import numpy
from tqdm.auto import tqdm
from multiprocessing import Pool

import emb_mgr


def mat_worker(arr):
    scalArr = numpy.dot(arr, numpy.transpose(arr))
    diego = numpy.diag(numpy.diag(scalArr))
    scalArrMod = scalArr - diego
    return scalArrMod

def sim_worker(args):
    i0, i1, mat0, mat1 = args
    nom = numpy.trace(
        numpy.dot(numpy.transpose(mat0),
                  mat1))
    denom1 = numpy.trace(
        numpy.dot(numpy.transpose(mat0),
                  mat0))
    denom2 = numpy.trace(
        numpy.dot(numpy.transpose(mat1),
                  mat1))
    Rv = nom / numpy.sqrt(denom1 * denom2)
    return i0, i1, Rv
    

def RV2coeff(dataList):
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
    for scalArrMod in tqdm(it, desc=" - Calculating scalArrMods", total=len(dataList)):
        scalArrList.append(scalArrMod)

    # Now compute the 'between study cosine matrix' C
    C = numpy.zeros((len(dataList), len(dataList)), numpy.float32)

    it = numpy.array(numpy.triu_indices(len(dataList), k=1)).T
    ita = ( (i0, i1, scalArrList[i0], scalArrList[i1]) for i0, i1 in it )
    itp = map(sim_worker, ita)
    for i0, i1, rv in tqdm(itp, desc=" - Calculating  Similarity", total=len(it)):
        C[i0, i1] = rv
        C[i1, i0] = rv

    return C


def get_arrays(series):
    return series.groupby("pid").apply(lambda x: numpy.vstack(x).T)

def take_multi(df, sl, level=0):
    return df.loc[vectors.index.levels[level][sl].tolist()]


def create_rv(mgr)


if __name__ == "__main__":
#     col = "d2v_v2"
#     vectors = pd.read_feather("vectors.feather")
#     vectors = vectors.set_index(["patient", "rid"])
    


    vec_name = "lda_d200_e10"
    mgr = emb_mgr.EmbMgr("emb_mgr/data")
    vectors = mgr.get_vectors(vec_name)
    
    sv = take_multi(vectors, slice(None, None))
    a = get_arrays(sv)

    sims = RV2coeff(a)
    numpy.save(f"managed/vectors/{vec_name}-RV.npy", sims)
    