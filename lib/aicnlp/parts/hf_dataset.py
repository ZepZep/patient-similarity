import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from sklearn.model_selection import StratifiedKFold

from datasets import Dataset
from transformers import AutoTokenizer

tqdm.pandas()


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        max_length=512,
        truncation=True,
    )


def make_train_test(relevant):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in skf.split(relevant, relevant.label):
        break

    ds = {
        "train": make_hf_dataset_part(relevant, train_index, tokenize_function),
        "test":  make_hf_dataset_part(relevant, test_index, tokenize_function),
    }
    return ds


def make_hf_dataset_part(relevant, indexer, tokenize_function, numproc=12):
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
        tokenize_function, batched=True, num_proc=numproc, desc="Tokenizing")
    ds = ds.remove_columns(['__index_level_0__', 'text'])
    return ds


def make_hf_dataset(hf_model="ufal/robeczech-base", out_path)
    parts = pd.read_feather(f"{out_path}/parts.feather")
    relevant = parts.query("label >= 0").reset_index(drop=True)

    print("--> Tokenizing")
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    ds = make_train_test(relevant)

    print("--> Saving dataset")
    ds["train"].save_to_disk(f"{out_path}/train.hf")
    ds["test"].save_to_disk(f"{out_path}/test.hf")
