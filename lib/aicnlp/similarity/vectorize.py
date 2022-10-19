from collections import namedtuple
from pathlib import Path
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

LSAResult = namedtuple("LSAModel", ["vectorizer", "decomposer"])


def get_methods():
    return {
        "Vtfi": (vectorize_tfidf, {}),
        "Vrbc": (vectorize_RobeCzech, {}),

    }


def tokenize(text):
    text = re.sub(r"[0-9]", "#", text)
    text = re.sub(r"([\.\,\:])(?!#)", r" \1 ", text)
    text = re.sub(r"\n", r" <br> ", text)

    return text.split()


def vectorize_tfidf(
    prefix, data_dir, input_file, dim,
    min_df=3, analyzer=tokenize,
    n_iter=5, random_state=42
    ):

    print(f"{prefix}loading", end="", flush=True)
    in_name = Path(input_file).with_suffix("").name
    records = pd.read_feather(input_file)


    print(f", TF-IDF", end="", flush=True)
    vectorizer = TfidfVectorizer(
        lowercase=False,
        min_df=min_df,
        analyzer=tokenize
    )
    x = vectorizer.fit_transform(records["text"])

    print(f", SVD", end="", flush=True)
    svd = TruncatedSVD(n_components=dim, n_iter=n_iter, random_state=random_state)
    vectors = svd.fit_transform(x)

    print(f", writing {len(records):7d} lines", end="", flush=True)
    records["vec"] = list(vectors)
    records.to_feather(f"{data_dir}/2/Vtfi{dim:03d}-{in_name}.feather")

    print(", DONE.", flush=True)
    # return vectors, LSAResult(vectorizer, svd)


def vectorize_RobeCzech(
    prefix, data_dir, input_file, dim=768,
    batch_size=16,
    n_iter=5, random_state=42,
    ):

    import transformers
    from transformers import AutoTokenizer, AutoModel
    import datasets
    from tqdm.auto import tqdm
    import torch
    from sklearn.decomposition import TruncatedSVD

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

    vectors = []
    dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, pin_memory=True)
    for batch in tqdm(dataloader, desc=f"{prefix} Embedding"):
        # batch.to("cuda")
        batch = {k: v.to("cuda") for k, v in batch.items()}
        pred = model(**batch).last_hidden_state[:,0,:].cpu().detach().numpy()
        vectors.append(pred)

    vectors = np.concatenate(vectors)
    # print(vectors.shape)
    print(prefix, end="", flush=True)

    if dim != vectors.shape[1]:
        print(f"SVD", end="", flush=True)
        svd = TruncatedSVD(n_components=dim, n_iter=n_iter, random_state=random_state)
        vectors = svd.fit_transform(vectors)
        print(f", ", end="", flush=True)


    print(f"writing {len(records):7d} lines", end="", flush=True)
    records["vec"] = list(vectors)
    records.to_feather(f"{data_dir}/2/Vrbc{dim:03d}-{in_name}.feather")

    print(", DONE.", flush=True)
