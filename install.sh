#!/bin/bash
# Preliminary installation script for the pipeline dependencies and environment

# Function to check if the script is running interactively
is_interactive() {
    [[ $- == *i* ]]
}

# Function to handle errors
handle_error() {
    if is_interactive; then
        return 1
    else
        exit 1
    fi
}

# This script is designed to be run on a fresh Ubuntu 22.04 installation with CUDA 12.2 and NVIDIA driver version 535.161.07.
# Static installation options
ENV_NAME="xprize_pipeline"
REQUIRED_PYTHON_VERSION="3.11.5"
UBUNTU_VERSION="22.04"
CUDA_VERSION="12.2"
NVIDIA_DRIVER_VERSION="535.161.07"

# Function to check OS and CUDA version
check_system_requirements() {
    # Check OS
    if [[ $(lsb_release -rs) != "$UBUNTU_VERSION" ]]; then
        echo "Warning: This script is designed for Ubuntu $UBUNTU_VERSION."
    fi

    # Check CUDA version
    if ! nvcc --version | grep "release $CUDA_VERSION" > /dev/null; then
        echo "Warning: This script is designed for CUDA $CUDA_VERSION."
    fi

    # Check NVIDIA driver version
    if ! nvidia-smi | grep "Driver Version: $NVIDIA_DRIVER_VERSION" > /dev/null; then
        echo "Warning: This script is designed for NVIDIA driver version $NVIDIA_DRIVER_VERSION."
    fi
}

if ! command -v micromamba &> /dev/null; then
    echo "micromamba could not be found. Please install micromamba before running this script."
    handle_error
fi

# Check system requirements
check_system_requirements

# Prepare environment
# Check if the environment exists
if micromamba env list | grep -Pq "^\s+$ENV_NAME\s+" > /dev/null; then
    echo "Environment $ENV_NAME already exists. Skipping creation."
    echo "WARNING: If the environment was created manually, the guarantees of this script are void. Use at your own risk."
else
    echo "Creating environment $ENV_NAME with Python $REQUIRED_PYTHON_VERSION."
    if ! micromamba create --name $ENV_NAME python="$REQUIRED_PYTHON_VERSION" -y -c conda-forge; then
        echo "Failed to create the environment $ENV_NAME with Python $REQUIRED_PYTHON_VERSION."
        handle_error
    fi
fi

micromamba activate "$ENV_NAME"

# Check Python version, could be an issue if the user has created the ENV_NAME micromamba environment manually beforehand with the wrong Python version 
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
if [[ "$PYTHON_VERSION" != "$REQUIRED_PYTHON_VERSION"* ]]; then
    echo \"Error: The environment \$ENV_NAME is using Python \$PYTHON_VERSION, but this script requires Python \$REQUIRED_PYTHON_VERSION.\"
    return
fi

# Install torch; assumes the system is using CUDA>=12.1
if [ ! "$(pip show torch)" ]; then
    if ! micromamba install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -c conda-forge; then
        echo "Failed to install torch."
        handle_error
    fi
else
    echo "torch already installed. Skipping installation."

fi

# Install gradio
if [ ! "$(pip show gradio)" ]; then
    if ! micromamba install gradio==4.29.0 -c conda-forge -y; then
        echo "Failed to install gradio."
        handle_error
    fi
else
    echo "gradio already installed. Skipping installation."
fi

# Install fastai
if [ ! "$(pip show fastai)" ]; then
    if ! micromamba install fastai -c fastai -c pytorch -c conda-forge -y; then
        echo "Failed to install fastai."
        handle_error
    fi
else
    echo "fastai already installed. Skipping installation."
fi

CWD=$(pwd)

# Install flat-bug; i.e. Multiple Object Detection and Segmentation
if [ ! "$(pip show flat-bug)" ]; then
    cd "$HOME"
    if [ ! -d "flat-bug" ]; then
        if ! git clone git@github.com:darsa-group/flat-bug.git; then
            echo "Failed to clone flat-bug repository."
            handle_error
        fi
    fi
    cd flat-bug
    git fetch
    git checkout dev_experiments
    git pull
    pip install -e .
else
    echo "flat-bug already installed. Skipping installation."
fi

# Install xprize_insectnet
if [ ! "$(pip show xprize_insectnet)" ]; then
    cd "$HOME"
    if [ ! -d "insectnet" ]; then
        if ! git clone git@github.com:GuillaumeMougeot/insectnet.git; then
            echo "Failed to clone insectnet repository."
            handle_error
        fi
    fi
    cd insectnet
    pip install -e .
else
    echo "xprize_insectnet already installed. Skipping installation."
fi

# Install rawpy
if [ ! "$(pip show rawpy)" ]; then
    cd $HOME
    # Check if libraw is installed; the command:
    # `pkg-config --exists --print-errors libraw` 
    # should return 0 if libraw is installed
    if [ ! "$(pkg-config --exists --print-errors libraw)" ]; then
        git clone git@github.com:LibRaw/LibRaw.git libraw
        git clone git@github.com:LibRaw/LibRaw-cmake.git libraw-cmake
        cd libraw
        git checkout 0.20.0
        cp -R ../libraw-cmake/* .
        cmake . -DCMAKE_INSTALL_PREFIX=$HOME/.local
        make
        make install
    fi
    if [ ! "$(pip show cython)" ]; then
        pip install cython
    fi
    pip install rawpy --user --no-deps
else
    echo "rawpy already installed. Skipping installation."
fi

# Return to the original directory
cd "$CWD"

# Install the Tracking/Clustering, Classification &  Localization submodules
. "$CWD"/sync_submodules.sh

# Deactivate the environment to restore the original environment
micromamba deactivate

# Print completion message
echo "Installation complete."
echo ""

# Print instructions
echo "INSTRUCTIONS:"
echo "  Please activate the environment before running the pipeline:"
echo "      'micromamba activate $ENV_NAME' "
echo "  You can deactivate the environment with:"
echo "      'micromamba deactivate'"
echo "  You can remove the environment with:"
echo "      'micromamba env remove -n $ENV_NAME'"
echo "  Run the pipeline:"
echo "      'python pipeline.py -i [<INPUT_IMAGES> ...] [-o <OUT_FOLDER>]'"
echo "  Start the web interface:"
echo "      'gradio app.py'"
