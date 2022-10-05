import pandas as pd
import numpy as np


# mtehods_base = {
#     "Fall": (RecordFilter, {"filter_setting": None}),
# }
# methods_recurrent = {
#     f"Fr{i:02d}": (RecordFilter, {"filter_setting": i})
#         for i in range(1, 11)
# }
#
# methods = mtehods_base | methods_recurrent

def get_methods(categories_file):
    categories = pd.read_feather(categories_file)
    filters = categories.columns[4:]
    methods = {
        f: (filter_records, {"categories_file": categories_file, "filter_setting": f})
        for f in filters
    }
    return methods


def filter_records(prefix, data_dir, input_file, categories_file, filter_setting):
    print(f"{prefix}loading", end="", flush=True)
    parts = pd.read_feather(input_file)
    categories = pd.read_feather(categories_file)

    print(", filtering", end="", flush=True)
    categories.set_index("index")
    relevant_categories = set(categories[categories[filter_setting]].index)
    old_names = "rid pid rord srord text".split()
    relevant_parts = parts[old_names][parts["pred"].isin(relevant_categories)]

    print(", merging", end="", flush=True)
    relevant_texts = relevant_parts.groupby(["pid", "rord"])["text"].apply("\n<br>\n".join)
    relevant_texts = relevant_texts.reset_index()

    print(f", writing {len(relevant_texts):7d} lines", end="", flush=True)
    relevant_texts.to_feather(f"{data_dir}/1/{filter_setting}.feather")

    print(", DONE.", flush=True)
    pn = parts["pid"].nunique()
    fn = relevant_texts["pid"].nunique()
    if pn != fn:
        print(f"{prefix}!!! only {fn}/{pn} have at least one record.")
