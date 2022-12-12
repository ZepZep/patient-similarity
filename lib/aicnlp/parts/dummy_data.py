import pandas as pd
import lorem
from random import choice, randint, random
from pathlib import Path

def get_segment():
    parts = []

    if randint(0,1):
        parts.append(get_title())
        if randint(0,1):
            parts.append(" ")
        else:
            parts.append("\n")
        parts.append(get_text(randint(20, 200)))
    elif randint(0,1):
        parts.append(get_text(randint(20, 200)))

    if randint(0,1):
        parts.append(get_text(randint(20, 200)))
        while randint(0,3):
            parts.append(" - " + get_text(randint(20, 200)) + "\n")

    if random() < 0.5:
        parts.append("\n")
    if random() < 0.25:
        parts.append("\n")
    return "".join(parts)

def get_record(size):
    return "".join(get_segment() for _ in range(size))

def get_text(size):
    return lorem.paragraph()[:size]

title_parts = [
    ["a", "A", "some", "other", "OTHER", None, None],
    " ",
    ["different", "OTHER", "same", "interestin", None, None, None],
    " ",
    ["title", "TITLE"],
    " ",
    "0123456789 ",
]
def get_title():
    choices = (choice(part) for part in title_parts)
    title = "".join(part for part in choices if part) + ":"
    return title

def make_rec(path, n_pac=100, n_rec=10, n_seg=6):
    records = []
    for pid in range(n_pac):
        for rord in range(randint(n_rec//2, n_rec*2)):
            records.append({
                "pid": pid,
                "rord": rord,
                "text": get_record(randint(n_seg//2, n_seg*2)),
            })

    rec = pd.DataFrame.from_records(records)
    rec.to_feather(path)

def make_pac(path, n_pac=100):
    pac = pd.DataFrame({
        "xml": [f"pac{i:05}.xml" for i in range(n_pac)],
        "id_pac": [1000000 + i for i in range(n_pac)],
    })
    pac.to_feather(path)


def ensure_mgr_data(path):
    rec_path = f"{path}/rec.feather"
    if not Path(rec_path).is_file():
        print(f"WARNING: could not find {repr(rec_path)}. Creating dummy data.")
        make_rec(rec_path, n_pac=100, n_rec=10, n_seg=6)

    pac_path = f"{path}/pac.feather"
    if not Path(pac_path).is_file():
        print(f"WARNING: could not find {repr(pac_path)}. Creating dummy data.")
        make_pac(pac_path, n_pac=100)



