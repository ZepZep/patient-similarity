import pandas as pd
import numpy as np
from scipy import stats
from pprint import pprint
from tqdm.auto import tqdm

from itertools import product, combinations
import importlib

from contextlib import redirect_stderr
import io
import sys
import os


AICOPE_PY_LIB = os.environ.get("AICOPE_PY_LIB")
if AICOPE_PY_LIB and AICOPE_PY_LIB not in sys.path: sys.path.append(AICOPE_PY_LIB)
import aicnlp
PACSIM_DATA = os.environ.get("AICOPE_SCRATCH") + "/pacsim"


corr_kendall = lambda x, y: stats.kendalltau(x, y, nan_policy='omit', alternative="greater")
corr_spearman = lambda x, y: stats.spearmanr(x, y, alternative="greater")
def corr_ks(x, y):
    ck, pk = corr_kendall(x,y)
    cs, ps = corr_spearman(x,y)
    return (ck+cs)/2, (pk+ps)/2

# corr_fcn = corr_spearman
corr_fcn = corr_kendall
# corr_fcn = corr_ks


def extract_annotations(valdata):
    annotations = []
    pivots = {x["pivot_patient"] for x in valdata}
    for pivot in pivots:
        relevant = {(f'a{x["user"]["id"]}', x["user"]["username"]):
                x for x in valdata if x["pivot_patient"] == pivot}
        for (ann, _), x in relevant.items():
            for proxy, vals in x["ratings"].items():
                for col, val in vals.items():
                    annotations.append({
                        "pivot": str(pivot),
                        "proxy": str(proxy),
                        "cat": f"{int(col):02d}",
                        "ann": ann,
                        "value": val,
                    })
    annotations = pd.DataFrame.from_records(annotations)

    # remove -1
    annotations.loc[annotations["value"] == -1, "value"] = None

    #normalized values
    sums = annotations.groupby(["pivot", "cat", "ann"])["value"].transform("sum")
    annotations["norm"] = annotations["value"] / sums
    # annotations.sort_values(["pivot", "cat", "ann"]).head(5)

    # annotations.head(2)
    mean_annotations = annotations.groupby(["pivot", "proxy", "cat"])["norm"].mean().reset_index()
    mean_annotations.rename(columns={"norm": "value"}, inplace=True)

    return annotations, mean_annotations


def matsim_row(matsim, id_pac, mgr):
    pid = mgr.tr["id_pac", "pid"][str(id_pac)]
    try:
        pac_row = list(matsim["patients"]).index(pid)
    except ValueError:
        return None
    return pac_row


def extract_predictions(annotations, matsims, mgr):
    predictions = []
    unknown = set()
    for cat, matsim, in matsims.items():
        for pivot, proxy in annotations.groupby(["pivot", "proxy"]).groups.keys():
            value = -100

            pivot_index = matsim_row(matsim, pivot, mgr)
            if pivot_index is None:
                value = None
                unknown.add((cat+"_pivot", pivot))

            proxy_index = matsim_row(matsim, proxy, mgr)
            if proxy_index is None:
                value = None
                unknown.add((cat+"_proxy", proxy))

            if value is not None:
                value = matsim["sim"][pivot_index, proxy_index]

            predictions.append({
                "pivot": pivot,
                "proxy": proxy,
                "cat": cat,
                "value": value
            })

    return pd.DataFrame.from_records(predictions), unknown

def ann_corr_mean(cors):
    cors = [x for x in cors if x==x]
    if not cors:
        return None
    return sum(cors) / len(cors)

def get_correlations_ann(annotations):
    correlations = []
    for (pivot, cat), batch in annotations.groupby(["pivot", "cat"])[["proxy", "ann", "value"]]:
        anns = batch["ann"].unique()
        cors = []
        for a1, a2 in combinations(anns, 2):
            s1 = batch[batch["ann"] == a1].sort_values("proxy")
            s2 = batch[batch["ann"] == a2].sort_values("proxy")
            cors.append(corr_fcn(s1["value"], s2["value"])[0])

        # c, p = corr_fcn(s1, s2_all, alternative="greater")
        correlations.append({
            "pivot": pivot,
            "cat": cat,
            "ctype": "ann",
            "value": ann_corr_mean(cors),
            "pval": 0
        })
    return correlations


def get_correlations_all(mean_annotations, predictions):
    correlations = []
    pred_dict = dict(list(predictions.groupby(["pivot", "cat"])[["proxy", "value"]]))
    for (pivot, cat), batch in mean_annotations.groupby(["pivot", "cat"])[["proxy", "value"]]:
        ## Possible error in ordering: [["proxy", "value"]]
        batch = batch.sort_values("proxy")
        s2_all = pred_dict[pivot, "all"].sort_values("proxy")
        s2_cat = pred_dict[pivot, cat].sort_values("proxy")

        # if batch["proxy"].tolist() != s2_all["proxy"].tolist():
        #     print(batch["proxy"].tolist())
        #     print(s2_all["proxy"].tolist())
        #     print()


        c, p = corr_fcn(batch["value"], s2_all["value"])
        correlations.append({
            "pivot": pivot,
            "cat": cat,
            "ctype": "all",
            "value": c,
            "pval": p,
        })
        c, p = corr_fcn(batch["value"], s2_cat["value"])
        correlations.append({
            "pivot": pivot,
            "cat": cat,
            "ctype": "cat",
            "value": c,
            "pval": p,
        })
    return correlations


def get_correlations(annotations, mean_annotations, predictions):
    correlations = []

    correlations.extend(get_correlations_ann(annotations))
    correlations.extend(get_correlations_all(mean_annotations, predictions))

    return pd.DataFrame.from_records(correlations)


def get_matsims(matmethod, vmethod):
    matsims = {}
    for i in range(1, 11):
        matsims[f"{i:02d}"] = np.load(f"{PACSIM_DATA}/3/{matmethod}-{vmethod}-Fr{i:02d}.npz")
    matsims["all"] = np.load(f"{PACSIM_DATA}/3/{matmethod}-{vmethod}-Fall.npz")
    return matsims


def get_data(matmethod, vmethod, valdata, mgr):
    matsims = get_matsims(matmethod, vmethod)
    annotations, mean_annotations = extract_annotations(valdata)
    predictions, unknown = extract_predictions(annotations, matsims, mgr)
    if len(unknown) > 10:
        print("unknown:")
        pprint(unknown)
    correlations = get_correlations(annotations, mean_annotations, predictions)

    return annotations, mean_annotations, predictions, correlations


def get_data_all(matmethods, vmethods, valdata, mgr):
    annotations, mean_annotations = extract_annotations(valdata)

    all_predictions = {}
    all_correlations = {}
    it = list(product(matmethods, vmethods))
    for matmethod, vmethod in tqdm(it):
        matsims = get_matsims(matmethod, vmethod)
        predictions, unknown = extract_predictions(annotations, matsims, mgr)
        all_predictions[matmethod, vmethod] = predictions
        if len(unknown) > 10:
            print("unknown:")
            pprint(unknown)

        with redirect_stderr(io.StringIO()) as f:
            correlations = get_correlations(annotations, mean_annotations, predictions)
        all_correlations[matmethod, vmethod] = correlations

    return annotations, mean_annotations, all_predictions, all_correlations


