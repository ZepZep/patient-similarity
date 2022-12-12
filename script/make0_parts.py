import os
import sys

AICOPE_PY_LIB = os.environ.get("AICOPE_PY_LIB")
if AICOPE_PY_LIB and AICOPE_PY_LIB not in sys.path: sys.path.append(AICOPE_PY_LIB)

PACSIM_DATA = os.environ.get("AICOPE_SCRATCH") + "/pacsim"

HF_MODEL = "ufal/robeczech-base"
PARTS_PATH = f"{PACSIM_DATA}/parts/"

os.makedirs(f"{PARTS_PATH}/predictions", exist_ok=True)
os.makedirs(f"{PARTS_PATH}/models", exist_ok=True)

def section(title, w=50):
    print("\n\n"+"#"*w+"\n"+f"{title:^{w}}\n"+"#"*w+"\n")


# Ensure data for manager
from aicnlp.parts.dummy_data import ensure_mgr_data
ensure_mgr_data(f"{PACSIM_DATA}/mgrdata")


# Load data with manager
from aicnlp import emb_mgr
mgr = emb_mgr.EmbMgr(f"{PACSIM_DATA}/mgrdata")


# Segment notes, extract and normalize titles
section("Extraction")
from aicnlp.parts.segments import make_segments
make_segments(
    parts_path=PARTS_PATH,
    records=mgr.rec,
)


# Train vector models, title similarity
section("Vectors")
from aicnlp.parts.vectors import train_vectors
train_vectors(
    parts_path=PARTS_PATH,
    dim=50,
    hidden_size=64,
)


# Create HuggingFace dataset
section("HF dataset")
from aicnlp.parts.hf_dataset import make_hf_dataset
make_hf_dataset(
    parts_path=PARTS_PATH,
    hf_model=HF_MODEL,
)


# Train Bi-LSTM model
section("Bi-LSTM")
from aicnlp.parts.bilstm import train_bilstm
train_bilstm(
    parts_path=PARTS_PATH,
    cut=150,
    dropout=0.1,
)


# Train RobeCzech model (transformer model)
## might need to change the batch size or other arguments
## to fit in your GPU in robeczech.py
section("RobeCzech")
from aicnlp.parts.robeczech import train_robeczech
train_robeczech(
    parts_path=PARTS_PATH,
    batch_size=2,
    hf_model=HF_MODEL,
)

