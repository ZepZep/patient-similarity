import json
import sys
import os

AICOPE_PY_LIB = os.environ.get("AICOPE_PY_LIB")
if AICOPE_PY_LIB and AICOPE_PY_LIB not in sys.path: sys.path.append(AICOPE_PY_LIB)
import aicnlp

PACSIM_DATA = os.environ.get("AICOPE_SCRATCH") + "/pacsim"


# Load mgr
from aicnlp import emb_mgr
mgr = emb_mgr.EmbMgr(f"{PACSIM_DATA}/mgrdata")


# start dashboard
from aicnlp.validation import similarity_dashboard

app = similarity_dashboard.get_app(
    mgr,
    name="simdash",
    lorem=True,
)

app.run_server(debug=True)
