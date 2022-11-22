from collections import namedtuple
from pathlib import Path
import pandas as pd
import numpy as np
import re
from tqdm.auto import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.callbacks import CallbackAny2Vec

tqdm.pandas()
LSAResult = namedtuple("LSAModel", ["vectorizer", "decomposer"])


def get_methods():
    return {
        "Vlsa": (vectorize_lsa, {}),
        "Vd2v": (vectorize_doc2vec, {}),
        "Vrbc": (vectorize_RobeCzech, {}),
    }


def tokenize(text):
    text = re.sub(r"[0-9]", "#", text)
    text = re.sub(r"([\.\,\:])(?!#)", r" \1 ", text)
    text = re.sub(r"\n", r" <br> ", text)

    return text.split()


def vectorize_lsa(
    prefix, data_dir, input_file, dims,
    min_df=3, tokenize_fcn=tokenize,
    n_iter=5, random_state=42
    ):

    print(f"{prefix}loading", end="", flush=True)
    in_name = Path(input_file).with_suffix("").name
    records = pd.read_feather(input_file)


    print(f", TF-IDF", end="", flush=True)
    vectorizer = TfidfVectorizer(
        lowercase=False,
        min_df=min_df,
        analyzer=tokenize_fcn
    )
    vectors_emb = vectorizer.fit_transform(records["text"])
    print(",")


    for dim in dims:
        print(prefix, end="", flush=True)
        if dim != vectors_emb.shape[1]:
            print(f"SVD {dim:03d}", end="", flush=True)
            svd = TruncatedSVD(n_components=dim, n_iter=n_iter, random_state=random_state)
            vectors_svd = svd.fit_transform(vectors_emb)
            print(f", ", end="", flush=True)
        else:
            vectors_svd = vectors_emb

        print(f"writing {len(records):7d} lines", end="", flush=True)
        records["vec"] = list(vectors_svd)
        records.to_feather(f"{data_dir}/2/Vlsa{dim:03d}-{in_name}.feather")
        print()

    print(f"{prefix}DONE.", flush=True)
    # return vectors, LSAResult(vectorizer, svd)


class TqdmProgress(CallbackAny2Vec):
    def __init__(self, total, inc=1, **kwargs):
        self.pbar = tqdm(total=total, **kwargs)
        self.inc = inc
    def on_epoch_end(self, model):
        self.pbar.update(self.inc)
    def on_train_end(self, model):
        self.pbar.close()



def vectorize_doc2vec(
    prefix, data_dir, input_file, dims,
    window=5, min_count=5, workers=4, epochs=10,
    tokenize_fcn=tokenize,
    ):

    print(f"{prefix}loading", end="", flush=True)
    in_name = Path(input_file).with_suffix("").name
    records = pd.read_feather(input_file)
    print(",")

    def make_tagged_document(row):
        tokens = tokenize_fcn(row["text"])
        return TaggedDocument(tokens, [f"{row.name}"])

    tqdm.pandas(desc=f'{prefix}Tokenizing')
    docs = records.progress_apply(make_tagged_document, axis=1)

    for dim in dims:
        tqdmcb = TqdmProgress(epochs, desc=f"{prefix}Training doc2vec {dim:03d}, epoch")
        model = Doc2Vec(
            docs.to_list(), vector_size=dim, window=window,
            min_count=min_count, workers=workers, epochs=epochs,
            callbacks=[tqdmcb]
        )

        print(f"{prefix}  writing {len(records):7d} lines", end="", flush=True)
        records["vec"] = list(model.dv.vectors)
        records.to_feather(f"{data_dir}/2/Vd2v{dim:03d}-{in_name}.feather")
        print()

    print(f"{prefix}DONE.", flush=True)

def vectorize_RobeCzech(
    prefix, data_dir, input_file, dims=[768],
    batch_size=16,
    n_iter=5, random_state=42,
    ):

    import transformers
    from transformers import AutoTokenizer, AutoModel
    import datasets
    import torch

    print(f"{prefix}loading data", end="", flush=True)
    in_name = Path(input_file).with_suffix("").name
    records = pd.read_feather(input_file)
    # records = records.head(100)

    print(f", loading model", end="", flush=True)

    tokenizer = AutoTokenizer.from_pretrained("ufal/robeczech-base")
    def tokenize_function(ds):
        return tokenizer(
            ds["text"],
            padding="max_length",
            max_length=512,
            truncation=True,
        )

    model = AutoModel.from_pretrained(
        f"{data_dir}/parts/checkpoint-180000/",
        output_loading_info=False,
    )
    model = model.half().cuda()

    print(",", flush=True)

    ds = datasets.Dataset.from_pandas(records)
    ds = ds.map(tokenize_function, batched=True,  desc=f"{prefix} Tokenizing")
    ds = ds.remove_columns(['pid', 'rord', 'text'])
    ds.set_format('torch')

    vectors_emb = []
    dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, pin_memory=True)
    for batch in tqdm(dataloader, desc=f"{prefix} Embedding"):
        # batch.to("cuda")
        batch = {k: v.to("cuda") for k, v in batch.items()}
        pred = model(**batch).last_hidden_state[:,0,:].cpu().detach().numpy()
        vectors_emb.append(pred)

    vectors_emb = np.concatenate(vectors_emb)

    for dim in dims:
        print(prefix, end="", flush=True)
        if dim != vectors_emb.shape[1]:
            print(f"SVD {dim:03d}", end="", flush=True)
            svd = TruncatedSVD(n_components=dim, n_iter=n_iter, random_state=random_state)
            vectors_svd = svd.fit_transform(vectors_emb)
            print(f", ", end="", flush=True)
        else:
            vectors_svd = vectors_emb

        print(f"writing {len(records):7d} lines", end="", flush=True)
        records["vec"] = list(vectors_svd)
        records.to_feather(f"{data_dir}/2/Vrbc{dim:03d}-{in_name}.feather")
        print()

    print(f"{prefix}DONE.", flush=True)



