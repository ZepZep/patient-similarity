import os
import sys

AICOPE_PY_LIB = os.environ.get("AICOPE_PY_LIB")
if AICOPE_PY_LIB and AICOPE_PY_LIB not in sys.path: sys.path.append(AICOPE_PY_LIB)
from aicnlp.similarity import RecordFilterComputer

PACSIM_DATA = os.environ.get("AICOPE_SCRATCH") + "/pacsim"


computer = RecordFilterComputer(PACSIM_DATA)
computer.calculate()


