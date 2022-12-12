import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from sklearn.model_selection import StratifiedKFold
from functools import partial

from datasets import Dataset
from transformers import AutoTokenizer

tqdm.pandas()


def make_train_test(relevant, tok_fcn):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in skf.split(relevant, relevant.label):
        break

    ds = {
        "train": make_hf_dataset_part(relevant, train_index, tok_fcn),
        "test":  make_hf_dataset_part(relevant, test_index, tok_fcn),
    }
    return ds


def make_hf_dataset_part(relevant, indexer, tok_fcn):
    df = pd.concat([
        pd.DataFrame({
            "text": relevant.text[indexer],
            "label": relevant.label[indexer],
        }),
        pd.DataFrame({
            "text": relevant.stext[indexer],
            "label": relevant.label[indexer],
        })
    ])
    
    ds = Dataset.from_pandas(df).map(
        tok_fcn, batched=True, desc="Tokenizing")
    ds = ds.remove_columns(['__index_level_0__', 'text'])
    return ds


# tok = AutoTokenizer.from_pretrained("ufal/robeczech-base")


def make_hf_dataset(parts_path, hf_model="ufal/robeczech-base"):
    parts = pd.read_feather(f"{parts_path}/parts.feather")
    relevant = parts.query("label >= 0").reset_index(drop=True)

    print("--> Tokenizing")
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    
    tok_fcn = lambda examples : tokenizer(
        examples["text"],
        padding="max_length",
        max_length=512,
        truncation=True
    )

    ds = make_train_test(relevant, tok_fcn)

    print("--> Saving dataset")
    ds["train"].save_to_disk(f"{parts_path}/train.hf")
    ds["test"].save_to_disk(f"{parts_path}/test.hf")
