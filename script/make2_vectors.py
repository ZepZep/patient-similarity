import os
import sys

AICOPE_PY_LIB = os.environ.get("AICOPE_PY_LIB")
if AICOPE_PY_LIB and AICOPE_PY_LIB not in sys.path: sys.path.append(AICOPE_PY_LIB)
from aicnlp.similarity import VectorizeComputer

PACSIM_DATA = os.environ.get("AICOPE_SCRATCH") + "/pacsim"


computer = VectorizeComputer(PACSIM_DATA)
computer.calculate(
    # patterns=["Fall.feather"],
    dim=50,
    min_df=3,
    n_iter=5,
    random_state=42
)
