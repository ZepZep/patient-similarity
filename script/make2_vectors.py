import os
import sys

AICOPE_PY_LIB = os.environ.get("AICOPE_PY_LIB")
if AICOPE_PY_LIB and AICOPE_PY_LIB not in sys.path: sys.path.append(AICOPE_PY_LIB)
from aicnlp.similarity import VectorizeComputer

PACSIM_DATA = os.environ.get("AICOPE_SCRATCH") + "/pacsim"
RECALCULATE = False


computer = VectorizeComputer(PACSIM_DATA)

if not RECALCULATE:
    computer.calculate(
        # patterns=["Fr01.feather"],
        methods=[
            ("Vd2v", {
                "dims": [50, 200],
                "min_count": 2,
                "workers": 4,
                "epochs": 40,
            }),
        ],
    )


else:
    computer.calculate(
        # patterns=["Fr*.feather"],
        methods=[
            ("Vtfi", {
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
                "random_state": 42,
            }),
        ],
    )
