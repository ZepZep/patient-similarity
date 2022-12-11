# Patient similarity based on unstructured clinical notes

This repository contains code that was used in a Masters thesis called ["Patient similarity based on unstructured clinical notes"](https://is.muni.cz/th/c9gln/) written by Petr Zelina.

## Environment variables
The scripts and notebooks expet two envoronment variables:
* `AICOPE_PY_LIB` which contains a path that points to the `lib` folder of this repository. It is used to append the python path.
* `AICOPE_SCRATCH` path where the data and various outputs should be stored and loaded from. It should contain a `pacsim` folder


## How to run
1. load data to emb_mgr
2. run the `sctript/make0_parts.py` script to create `pacsim/parts/parts.feather` and `pacsim/parts/titles.feather` as well as the title embedding (`pacsim/parts/d2v_titles.npy`).
2. use the `notebook/parts_prediction.ipynb` notebook to predict titles for all segments, resulting in `pacsim/parts/parts_pred.feather`
3. use the `notebook/parts_clustering.ipynb` notebook to sort segments types into similarity categories, resulting in `pacsim/parts/categories_pred.feather`
4. configure and run patient similarity scripts
    * `script/make1_filtered_records.py`
    * `script/make2_vectors.py`
    * `script/make3_matsim.py`
5. run evaluation notebooks
    * `notebook/??` for segment classification statistics
    * `notebook/vis_general.ipynb` for visualisations of filtering statistics, matrix similarity methods comparison, and vectorization timing statistics
    * `notebook/vis_agreement.ipynb` for inter-annotator agreement results
    * `notebook/vis_validation.ipynb` for validation results
    * `notebook/vis_ablation.ipynb` for vectorization methods ablation results
6. dashboards
    * todo


## Docker
We include a dockerfile `docker/Dockerfile` for easier enviroment setup.
Example of build and run commands are in the `docker/commands.txt` file.


## Adapting for different datasets / languages
The filtering step contains some dataset-specific choices. These can be customized.
* the segmentation function probably needs to be tweaked to fit your dataset formating (`cut_record` function inside `lib/aicnlp/parts/segments.py`)
* title normalization function might need to be tweaked for some languages (`normalize_title` function inside `lib/aicnlp/parts/segments.py`)
* you may want to use different tokenization function for the vector methods (`tokenize_doc` function in `lib/aicnlp/parts/vectors.py`)
* you may want to use different HuggingFace transformer model and tokenizer. Change the `HF_MODEL` in `script/make0_parts.py`


## Interactive visualisation
We use the Tensorboard embedding projector to visualise the vector space of the 2078 extracted titles. It is available [here](https://zepzep.github.io/clinical-notes-extraction/pages/projector/).

The bookmarks (bottom right) contain 3 presets:
* results from clustering
* neighbours of the comorbidities title
* neighbours of the medication title

All of them use 2D T-SNE dimensionality reduction.
