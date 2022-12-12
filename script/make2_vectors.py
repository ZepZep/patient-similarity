import os
import sys
from itertools import product

AICOPE_PY_LIB = os.environ.get("AICOPE_PY_LIB")
if AICOPE_PY_LIB and AICOPE_PY_LIB not in sys.path: sys.path.append(AICOPE_PY_LIB)
from aicnlp.similarity import VectorizeComputer

PACSIM_DATA = os.environ.get("AICOPE_SCRATCH") + "/pacsim"

# PRESET = "custom"
PRESET = "recalculate"
# PRESET = "ablation"

computer = VectorizeComputer(PACSIM_DATA)

if PRESET == "custom":
    computer.calculate(
        # patterns=["Fr01.feather"],
        methods=[
            ("Vlsa", {
                "dims": [50, 200],
                "min_df": 1,
                "n_iter": 20,
                "random_state": 42,
            }),
        ],
    )
elif PRESET == "recalculate":
    computer.calculate(
        # patterns=["Fr*.feather"],
        methods=[
            ("Vlsa", {
                "dims": [50, 200],
                "min_df": 1,
                "n_iter": 20,
                "random_state": 42,
            }),
            ("Vd2v", {
                "dims": [50, 200],
                "min_count": 5,
                "workers": 4,
                "epochs": 40,
            }),
            ("Vrbc", {
                "dims": [50, 200],
                "n_iter": 10,
                "batch_size": 2,
                "random_state": 42,
            }),
        ],
    )
elif PRESET == "ablation":
    ablmethods = [
        ("Vlsa", {
            "dims": [[50]],
            "min_df": [1, 2, 3],
            "n_iter": [10, 20, 30],
            "random_state": [42],
            "ablation": [True]
        }),
        ("Vd2v", {
            "dims": [[50]],
            "min_count": [3,5,7],
            "window": [3,5,7],
            "workers": [4],
            "epochs": [30, 40, 50],
            "ablation": [True],
            "ablid": [1,2,3,4]
        }),
        ("Vrbc", {
            "dims": [[50]],
            "n_iter": [10],
            "random_state": [42],
            "finetuned": [True, False],
            "ablation": [True],
        }),
        ("Vrbc", {
            "dims": [[50]],
            "n_iter": [10, 20, 30],
            "random_state": [42],
            "finetuned": [True],
            "ablation": [True],
        }),
    ]

    def expand_ablation(ablmethods):
        out = []
        for method, pardict in ablmethods:
            for values in product(*pardict.values()):
                tuples = zip(pardict.keys(), values)
                out.append((method, dict(tuples)))
        return out

    from pprint import pprint

    methods = expand_ablation(ablmethods)

    computer.calculate(
        patterns=["Fall.feather"],
        methods=methods
    )
else:
    print(f"Unknown preset {repr(PRESET)}")
    exit(1)
