# Patient similarity based on unstructured clinical notes

This repository contains code that was used in a Masters thesis called ["Patient similarity based on unstructured clinical notes"](https://is.muni.cz/th/c9gln/) written by Petr Zelina.


## Environment variables
The scripts and notebooks expcet two envoronment variables:
* `AICOPE_PY_LIB` which contains a path that points to the `lib` folder of this repository. It is used to append the python path.
* `AICOPE_SCRATCH` path where the data and various outputs should be stored and loaded from. It should contain a `pacsim` folder and other subfolders
```
mkdir scratch scratch/pacsim scratch/pacsim/parts scratch/pacsim/1 scratch/pacsim/2 scratch/pacsim/3 scratch/pacsim/simvis scratch/pacsim/mgrdata
```


## Docker
We include a dockerfile `docker/Dockerfile` for easier enviroment setup.
Example of build and run commands are in the `docker/commands.txt` file.

You can also pull a pre-built docker image from [DockerHub](https://hub.docker.com/r/zepzep/patient_similarity).

You need [Nvidia docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU support.

The docker image starts a Jupyter server on [http://localhost:4444/](http://localhost:4444/). You can connect to it with your browser and use the notebooks or the terminal. The password is `password`. Alternatively, you `docker exec` bash in the container.

```
# build docker image
cd docker
docker build --network host -t zepzep/patient_similarity .

# or pull docker image
docker pull zepzep/patient_similarity

# run docker container
docker run --gpus all -it --network host --name pacsim zepzep/patient_similarity

# connect to running container with bash
docker exec -it -u 1000 pacsim /bin/bash
```


## How to run
1. load data to `emb_mgr`
    * to make it easier, we added a dummy data generator that automatically generates dummy data for the `emb_mgr` if you run the `sctript/make0_parts.py` without manually initializing the `emb_mgr`
    * If you want to add your own data create 2 files in the `pacsim/mgrdata/` folder
        * `rec.feather` containing the records. It should be a DataFrame with columns `pid` (patient id), `rord` (index of record of each patient separately), and `text` (text of the record). `(pid, rord)` should be a unique key.)
        * `pac.feather` containing details about patients. It should be a DataFrame with columns `xml` (name of the patient xml file, used for translation of IDs) and `id_pac` (internal hospital id, used for translation of IDs). This can be mostly random, but some dashboards use it.
        * you can look at `lib/aicnlp/parts/dummy_data.py` to see how the dummy data is generated
2. run the `sctript/make0_parts.py` script to create
    * `pacsim/parts/parts.feather` - segmented patient records
    * `pacsim/parts/titles.feather` - extracted titles
    * `pacsim/parts/d2v_titles.npy` - title embedding for segment type similarity.
    * `pacsim/parts/tid2t.pickle` - dictionary title_id -> title_text
    * The RobeCzech classifier might take a long time to train and possibly needs to have some parameters adjusted (like batch_size, gradient accumulation, ...) You can terminate it early, in which case the `Vrbc` vectorizer will use the uninitialized `manual_save`
2. use the `notebook/parts_prediction.ipynb` notebook to predict titles for all segments, resulting in `pacsim/parts/parts_pred.feather`
3. use the `notebook/parts_clustering.ipynb` notebook to sort segment types into similarity categories, resulting in `pacsim/parts/categories_pred.feather`. You can skip creating custom categories and just create non-filtered notes (`Fall`)
4. configure and run patient similarity scripts (the default presets create non-filtered variants of all methods in the main grid-search (1 * 6 * 3))
    * `script/make1_filtered_records.py`
    * `script/make2_vectors.py` (might need to turn off `notebook/parts_prediction.ipynb` to free up GPU memory for Vrbc vectorization)
    * `script/make3_matsim.py`
5. run evaluation notebooks
    * `notebook/vis_general.ipynb` for visualizations of filtering statistics, matrix similarity methods comparison, and vectorization timing statistics
    * `notebook/vis_agreement.ipynb` for inter-annotator agreement results
    * `notebook/vis_validation.ipynb` for grid-search and validation results
    * `notebook/vis_ablation.ipynb` for vectorization methods ablation results
6. dashboards
    * `dashboard/start_validation_dashboard.py` starts the Dash-based validation dashboard. It is useful for inspecting the interaction between filtering, vectorization and matrix similarity methods.
    * `dashboard/start_similarity_dashboard.py` starts the Dash-based similarity dashboard. It is useful for viewing similar patients.


## Adapting for different datasets / languages
The filtering step contains some dataset-specific choices. These can be customized.
* the segmentation function probably needs to be tweaked to fit your dataset formating (`cut_record` function inside `lib/aicnlp/parts/segments.py`)
* title normalization function might need to be tweaked for some languages (`normalize_title` function inside `lib/aicnlp/parts/segments.py`)
* you may want to use different tokenization function for the vector methods (`tokenize_doc` function in `lib/aicnlp/parts/vectors.py`)
* you may want to use different HuggingFace transformer model and tokenizer. Change the `HF_MODEL` in `script/make0_parts.py`


## Interactive visualization
We use the Tensorboard embedding projector to visualise the vector space of the 2078 extracted titles. It is available [here](https://zepzep.github.io/clinical-notes-extraction/pages/projector/).

The bookmarks (bottom right) contain 3 presets:
* results from clustering
* neighbours of the comorbidities title
* neighbours of the medication title

All of them use 2D T-SNE dimensionality reduction.
