import glob

import aicnlp.similarity.filter
import aicnlp.similarity.vectorize
import aicnlp.similarity.matsim

class AbstractComputer:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.all_methods = {}
        self.prec = 2

    def _calculeteIO(self, methods=None, patterns=None):
        if methods is None:
            methods_out = [(method, {}) for method in self.all_methods.keys()]
        else:
            methods_out = []
            for (method, cfg) in methods:
                if method not in self.all_methods:
                    print(f"Unknown method {repr(method)}")
                else:
                    methods_out.append((method, cfg))

        return methods_out, self._resolve_patterns(patterns)

    def _resolve_patterns(self, patterns):
        raise NotImplementedError()

    def calculate(self, methods=None, patterns=None):
        method_defs, inputs = self._calculeteIO(methods, patterns)
        for idi, input_file in enumerate(inputs):
            i_prog = f"[{idi+1:0{self.prec}d}/{len(inputs):0{self.prec}d}]"
            print(f"{i_prog} Working on input file {input_file}")
            for idm, (mn, mcfg) in enumerate(method_defs):
                m_prog = f"[{idm+1:0{self.prec}d}/{len(method_defs):0{self.prec}d}]"
                print(f"  {m_prog} Working on method {mn}")
                method_fcn, base_kwargs = self.all_methods[mn]
                method_kwargs = base_kwargs.copy()
                method_kwargs.update(mcfg)
                output = method_fcn(
                    prefix=" "*4,
                    data_dir=self.data_dir,
                    input_file=input_file,
                    **method_kwargs,
                )


class RecordFilterComputer(AbstractComputer):
    def __init__(self, data_dir, categories_file=None):
        super().__init__(data_dir)
        if categories_file is None:
            self.categories_file = "parts/categories_pred.feather"
        else:
            self.categories_file = categories_file
        self.parts_file = "parts/parts_pred.feather"
        self.all_methods = aicnlp.similarity.filter.get_methods(
            f"{data_dir}/{self.categories_file}"
        )

    def _resolve_patterns(self, patterns):
        if patterns is None:
            patterns = [f"{self.data_dir}/{self.parts_file}"]
        return patterns


class VectorizeComputer(AbstractComputer):
    def __init__(self, data_dir):
        super().__init__(data_dir)
        self.all_methods = aicnlp.similarity.vectorize.get_methods()

    def _resolve_patterns(self, patterns):
        if patterns is None:
            out = glob.glob(f"{self.data_dir}/1/F*.feather")
        else:
            out = []
            for p in patterns:
                out.extend(glob.glob(f"{self.data_dir}/1/{p}"))
        out.sort()
        return out


class MatsimComputer(AbstractComputer):
    def __init__(self, data_dir):
        super().__init__(data_dir)
        self.all_methods = aicnlp.similarity.matsim.get_methods()

    def _resolve_patterns(self, patterns):
        if patterns is None:
            out = glob.glob(f"{self.data_dir}/2/V*.feather")
        else:
            out = []
            for p in patterns:
                out.extend(glob.glob(f"{self.data_dir}/2/{p}"))
        out.sort()
        return out

