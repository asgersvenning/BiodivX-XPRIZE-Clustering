*There is no HuggingFace Space description here, as it is created by the GitHub Action which is triggered by the push event on the main branch.*

# BiodivX-XPRIZE-ML-pipeline
Simple repository for managing the overall CV ML execution pipeline of the ETH BiodivX for the XPRIZE Rainforest competition finals.

# Installation
The installation will **only** be verified for `Ubuntu 22.04` with `CUDA 12.2` with NVIDIA driver version `535.161.07`. The system must be pre-installed with `Python 3.11.5` and `micromamba`.

```bash
. install.sh
```

# Execution
## Activate environment

```bash
micromamba activate xprize_pipeline
cd "$HOME/BiodivX-XPRIZE-ML-pipeline"
```

## Run pipeline
Input should be path(s), director(y/ies) or glob(s) to such. If it is a .txt file, then it should contain lines corresponding to the former. Outputs will be saved in the output directory.

```bash
python pipeline.py [-h] -i [<INPUT_IMAGE_GLOB_DIR> ...] [-o <OUTPUT_DIR>]
```

**Example:**
```bash
python pipeline.py --input example_image1.jpg example_image2.jpg --output test_output_directory
```

# Output
The output contains various intermediate files and results, along with the final summary CSV called `track_metrics.csv` as well as the cropped instances sorted into folders named after the predicted class (some sanitizing is performed, most notably spaces are replaced by underscores).

The intermediate results for each submodule is stored in a folder named after the submodule i.e. `"localization"`/`"classification"`/"`tracking`".

This folder is identical to the ZIP-folder returned by the HuggingFace Gradio app.
