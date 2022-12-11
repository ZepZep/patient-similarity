import os
import sys

AICOPE_PY_LIB = os.environ.get("AICOPE_PY_LIB")
if AICOPE_PY_LIB and AICOPE_PY_LIB not in sys.path: sys.path.append(AICOPE_PY_LIB)

PACSIM_DATA = os.environ.get("AICOPE_SCRATCH") + "/pacsim"

HF_MODEL = "ufal/robeczech-base"
PARTS_PATH = f"{PACSIM_DATA}/parts/"

os.mkdir(f"{PARTS_PATH}/predictions")
os.mkdir(f"{PARTS_PATH}/models")


# Load data with manager
from aicnlp import emb_mgr
mgr = emb_mgr.EmbMgr(f"{PACSIM_DATA}/mgrdata")


# Segment notes, extract and normalize titles
from aicnlp.parts.segments import make_segments
make_segments(
    records=mgr.rec,
    out_path=PARTS_PATH
)


# Train vector models, title similarity
from aicnlp.parts.vectors import train_vectors
train_vectors(
    dim=50,
    hidden_size=64,
    out_path=PARTS_PATH,
)


# Create HuggingFace dataset
from aicnlp.parts.hf_dataset import make_hf_dataset
make_hf_dataset(
    hf_model=HF_MODEL,
    out_path=PARTS_PATH,
)


# Train Bi-LSTM model
from aicnlp.parts.bilstm import train_bilstm
train_bilstm(
    cut=150,
    dropout=0.1,
    out_path=PARTS_PATH,
)


# Train RobeCzech model (transformer model)
from aicnlp.parts.robeczech import train_robeczech
train_robeczech(
    hf_model=HF_MODEL,
    out_path=PARTS_PATH,
)

