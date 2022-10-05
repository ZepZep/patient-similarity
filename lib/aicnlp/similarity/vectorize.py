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
