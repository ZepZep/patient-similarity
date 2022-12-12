import os
import sys

AICOPE_PY_LIB = os.environ.get("AICOPE_PY_LIB")
if AICOPE_PY_LIB and AICOPE_PY_LIB not in sys.path: sys.path.append(AICOPE_PY_LIB)
from aicnlp.similarity import RecordFilterComputer

PACSIM_DATA = os.environ.get("AICOPE_SCRATCH") + "/pacsim"

# PRESET = "default"
PRESET = "Fall"
# PRESET = "custom"

computer = RecordFilterComputer(PACSIM_DATA)

if PRESET == "default":
    ## default configuration (Fall, Fr01, ... Fr10)
    computer.calculate()

elif PRESET == "Fall":
    ## only Fall
    computer.calculate(
        methods=[
            ("Fall", {})
        ],
    )

elif PRESET == "custom":
    ## custom columns
    computer.calculate(
        methods=[
            ("Fall", {}),
            ("Fr01", {}),
            ("Any other column from categories_pred.feather", {}),
        ],
    )

else:
    print(f"Unknown preset {repr(PRESET)}")
    exit(1)


