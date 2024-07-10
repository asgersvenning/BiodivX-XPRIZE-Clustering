*There is no HuggingFace Space description here, as it is created by the GitHub Action which is triggered by the push event on the main branch.*

# BiodivX-XPRIZE-Clustering
Simple repository for managing the clustering of images of individual arthropods for ETH BiodivX for the XPRIZE Rainforest competition finals.

# Installation
The installation will **only** be verified for `Ubuntu 22.04` with `CUDA 12.2` with NVIDIA driver version `535.161.07`. The system must be pre-installed with `Python 3.11.5` and `micromamba`.

```bash
. install.sh
```

# Execution
## Activate environment

```bash
micromamba activate xprize_clustering
cd "$HOME/BiodivX-XPRIZE-Clustering"
```

## Run clustering
Input should be a path or glob pattern to the image(s) to cluster. If it is a .txt file, then it should contain lines corresponding to the former.

```bash
python cluster.py [-h] --input/-i <INPUT_IMAGES> [<INPUT_IMAGES> ...] --out_folder/-o <OUTPUT_FOLDER> [--meta_folder/-mf <META_FOLDER>] [--time_format <TIME_FORMAT>] [--device/-d <DEVICE>] [--embed/-e <EMBED>] [--from_pretrained/-fp <FROM_PRETRAINED>] [--dedup/-kp] [--gen_cluster_dirs/-gs] [--cluster/-c <CLUSTER>] [--threshold/-th <THRESHOLD>] [--save_embs/-se] [--save_labels/-sl] [--num_classes/-x <NUM_CLASSES>] 
```

Details of command options:
```bash
-i INPUT [INPUT ...], --input INPUT [INPUT ...]
                    Path(s), director(y/ies) or glob(s) to such. If it is a .txt file, then it should contain lines corresponding to the former. Outputs
                    will be saved in the output directory.
-o OUT_FOLDER, --out_folder OUT_FOLDER
                    (Optional) Output folder for the clustered images.
-mf META_FOLDER, --meta_folder META_FOLDER
                    (Optional) Path of the metadata folder of the images. Useful for feature extraction.
--time_format TIME_FORMAT
                    Format for the time stamp in image file names.
-d DEVICE, --device DEVICE
                    (Optional) Device used to run the embedding model. Defaults to cuda:0 if available, else 'cpu'.
-e EMBED, --embed EMBED
                    (Optional) Name of the embedding method. Valid names: dict_keys(['dino', 'hierarchical'])
-fp FROM_PRETRAINED, --from_pretrained FROM_PRETRAINED
                    (Optional) HuggingFace model for embedding computation. Only required if 'embed=dino'.
-kp, --dedup        (Optional) Whether to deduplicate the images. If True, then all images will be stored in subfolders.
-gs, --gen_cluster_dirs
                    (Optional) Whether to generate an output folder with one subfolder for each cluster. Outliers are left aside.
-c CLUSTER, --cluster CLUSTER
                    (Optional) Name of the clustering method. Valid names: dict_keys(['cosine'])
-th THRESHOLD, --threshold THRESHOLD
                    (Optional) Threshold for the cosine similarity clustering.
-se, --save_embs    (Optional) Whether to save the embeddings in the output file.
-x NUM_CLASSES, --num_classes NUM_CLASSES
                    (Optional) Number of classes of the labels. Required for the evaluation
-sl, --save_labels  (Optional) Whether to save the labels in the output file. Labels must exit.
```

**Examples:**

Simple clustering with only an input folder of crops.

```bash
python cluster.py -i example/crops -o example/results
```

Clustering using both crops and metadata folders. In this example the time format had to be changed from the default `"%YYYY-%mm-%dd_%HH-%MM-%SS"` value.

```bash
python cluster.py -i example/crops -o example/results -mf example/metadata --time_format "%Y%m%d%H%M%S"
```

Clustering and creating a folder of crops per cluster. The threshold of the clustering method also changed to 0.8 (instead of 0.9, the default) to make it less stringent.

```bash
python cluster.py -i example/crops -o example/results -mf example/metadata --time_format "%Y%m%d%H%M%S" --gen_cluster_dirs -th 0.8
```

# Outputs

* A pkl containing the embeddings of the crops. 
* A CSV with two columns. `filename` with image filenames, `cluster` with cluster index.
* Optionally a folder named `clusters` if the `--gen_cluster_dirs` command with used.