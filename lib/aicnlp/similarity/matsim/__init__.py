from pathlib import Path
import pandas as pd
import numpy as np

from aicnlp.similarity.matsim import rv2, mms, eds


def get_methods():
    methods = {
        "rv2": rv2.calculate,
        "mms": mms.calculate,
        "eds": eds.calculate,
    }
    out = {}
    for method, fcn in methods.items():
        out[f"R{method}"] = (fcn, {"id_letter": "R"})
        out[f"M{method}"] = (fcn, {"patients": None, "id_letter": "M"})
    return out

#
# def resolve_relevance(fcn):
