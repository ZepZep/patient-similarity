import os
import sys

AICOPE_PY_LIB = os.environ.get("AICOPE_PY_LIB")
if AICOPE_PY_LIB and AICOPE_PY_LIB not in sys.path: sys.path.append(AICOPE_PY_LIB)
from aicnlp.similarity import VectorizeComputer

PACSIM_DATA = os.environ.get("AICOPE_SCRATCH") + "/pacsim"


computer = VectorizeComputer(PACSIM_DATA)
# computer.calculate(
#     # patterns=["Fall.feather"],
#     dim=200,
#     min_df=1,
#     n_iter=20,
#     random_state=42
# )

computer.calculate(
    patterns=["Fall.feather"],
    methods=["Vrbc"],
    dim=50,
    n_iter=10,
    random_state=42
)
