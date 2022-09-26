import pandas as pd
import numpy as np
import glob
from pathlib import Path
from itertools import product


class MethodAlreadyExists(Exception):
    pass

class EmbMgr:
    def __init__(self, path="/home/ubuntu/petr/mgrdata"):
        self.path = path
        self.pac = pd.read_feather(f"{path}/pac.feather")
        self.pac.index.name = "pid"

        self.tr = self.get_tr()

        self.rec = pd.read_feather(f"{path}/rec.feather")
        self.rec.index.name = "rid"
        self.rec = self.rec.set_index(["pid", "rord"], append=True)

        self.vectors = self.read_vectors(f"{path}/vectors")
        self.sims = self.read_sims(f"{path}/vectors")

    def get_tr(self):
        keys = {}
        keys["pid"] = self.pac.index
        for col in self.pac:
            keys[col] = self.pac[col]

        tr = {}
        for f, t in product(keys.keys(), keys.keys()):
            if f == t:
                continue
            tr[f, t] = dict(zip(keys[f], keys[t]))

        return tr


    def read_vectors(self, path):
        names = [Path(x).with_suffix("").name
                 for x in sorted(glob.glob(f"{path}/*.feather"))]
        return {name: None for name in names}

    def read_sims(self, path):
        names = [Path(x).with_suffix("").name
                 for x in sorted(glob.glob(f"{path}/*.npy"))]
        return {tuple(name.split("-")): None for name in names}

    # def get_pac_indexer(self, value, key="xml"):
    #     pid = value
    #     if key != "pid":
    #         pid = self.tr[key, "pid"][value]
    #     return self.rec["pid"] == pid

    def get_vectors(self, vmethod):
        if vmethod not in self.vectors:
            raise KeyError(f"vmethod {repr(vmethod)} not available")

        if self.vectors[vmethod] is None:
            df =  pd.read_feather(f"{self.path}/vectors/{vmethod}.feather")
            df.index = self.rec.index
            s = df.iloc[:,0]
            self.vectors[vmethod] = s

        return self.vectors[vmethod]

    def get_pac_vectors(self, vmethod, value, key="xml"):
        pid = value
        if key != "pid":
            pid = self.tr[key, "pid"][value]
        vec = self.get_vectors(vmethod)
        return vec.loc[:, pid,:]

#     def get_rid(self, record, patient, key="xml"):
#         pass

    def get_sims(self, vmethod, smethod):
        if vmethod not in self.vectors:
            raise KeyError(f"vmethod {repr(vmethod)} not available")

        key = (vmethod, smethod)

        if key not in self.sims:
            raise KeyError(f"combination {repr(key)} not available")

        if self.sims[key] is None:
            self.sims[key] = np.load(f"{self.path}/vectors/{vmethod}-{smethod}.npy")

        return self.sims[key]

    def get_dropdown_dicts(self, vmethod=None):
        if vmethod is None:
            return [
                {'label': key, 'value': key, 'disabled': False}
                for key in self.vectors.keys()
            ]

        if vmethod not in self.vectors:
            raise KeyError(f"vmethod {repr(vmethod)} not available")

        avail = [ s for v, s in self.sims.keys() if v == vmethod ]
        return [
            {'label': key, 'value': key, 'disabled': False}
            for key in avail
        ]

    def save_vectors(self, vmethod, df):
        pass

    def save_sims(self, vmethod, smethod, arr):
        pass

    def create_vectors(self, vmethod, vectorizer, overwrite=False):
        if (not overwrite) and (vmethod in self.vectors):
            raise MethodAlreadyExists(f"vmethod {repr(vmethod)} already exists.")
        x = vectorizer.fit(self)
        vdf = pd.DataFrame(
            {"vectors": list(x)},
            index=range(vectors.shape[0]))
        vdf.to_feather(f"{self.path}/vectors/{vmethod}.feather")
        self.vectors["vmethod"]

    def create_sims(self, vmethod, smethod, simmilator, overwrite=False):
        if (not overwrite) and ((vmethod, smethod) in self.sims):
            raise MethodAlreadyExists(f"method {repr(vmethod, smethod)} already exists.")

        pass

