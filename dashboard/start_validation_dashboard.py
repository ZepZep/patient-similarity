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


# Load validation data
from aicnlp.validation.agreement import get_data_all

with open("/home/ubuntu/petr/similarity/validace/evaluation-response-20220907.json", encoding='utf-8') as f:
    valdata = json.load(f)

matmethods = ["Rrv2", "Rmms", "Reds"]
vmethods = ["Vlsa050", "Vlsa200", "Vrbc050", "Vrbc200", "Vd2v050", "Vd2v200"]

# matmethods = [ "Rmms",]
# vmethods = ["Vrbc200"]

annotations, mean_annotations, all_predictions, all_correlations = get_data_all(matmethods, vmethods, valdata, mgr)


# start dashboard
from aicnlp.validation import validation_dashboard

app = validation_dashboard.get_app(
    mgr, mean_annotations, all_predictions, all_correlations,
    name="valdash",
    lorem=True,
)

app.run_server(debug=True)
