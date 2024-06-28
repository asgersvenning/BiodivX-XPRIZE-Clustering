# Initialize the Dockerfile with CUDA 12.2.0 and Ubuntu 22.04 as the base image
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Install necessary packages including openssh-client, lsb-release, and other dependencies
RUN apt-get update && apt-get install -y wget bzip2 curl ca-certificates openssh-client lsb-release git dos2unix openssl libgl1 libglib2.0-0 cmake && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the DEBIAN_FRONTEND to noninteractive to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt install -y python3.11 python3-pip git-all tzdata && \
    ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

# Update pip to the latest version
RUN python3.11 -m pip install --upgrade pip

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the user while doing secret setup
USER root

# Print openssl version
RUN openssl version

# Add the encrypted SSH key to the container
COPY id_rsa.enc /home/user/.ssh/id_rsa.enc

# Add the secret passphrase for decryption
RUN --mount=type=secret,id=EncryptionPassphrase \
    mkdir -p /home/user/.ssh && \
    cat /run/secrets/EncryptionPassphrase | tr -d '\r' > /home/user/.ssh/passphrase

# Add SSH key secret and perform necessary setup as root user
RUN openssl enc -aes-256-cbc -d -in /home/user/.ssh/id_rsa.enc -out /home/user/.ssh/id_rsa -pass file:/home/user/.ssh/passphrase && \
    chmod 600 /home/user/.ssh/id_rsa

# Generate the public key
RUN ssh-keygen -y -f /home/user/.ssh/id_rsa > /home/user/.ssh/id_rsa.pub && \
    chmod 644 /home/user/.ssh/id_rsa.pub

# Add GitHub to known hosts
RUN ssh-keyscan github.com >> /home/user/.ssh/known_hosts && \
    chown -R user:user /home/user/.ssh

# Create SSH config
RUN echo "Host *\n    StrictHostKeyChecking no\n" > /home/user/.ssh/config && \
    chmod 600 /home/user/.ssh/config

# Debug: Load the expected SHA256SUM of the SSH key
ARG GithubTokenSHA256SUM

# Debug: Print the hashes of the SSH key to verify its integrity
RUN sha256sum /home/user/.ssh/id_rsa

# Debug: Check the hashes match the expected value
RUN test "$(sha256sum /home/user/.ssh/id_rsa | cut -d ' ' -f 1)" = "$GithubTokenSHA256SUM" && echo "Secrets match" || echo "Secrets do not match"

# Switch to the user account
USER user

# Print python version
RUN which python3.11 && python3.11 --version

# Pre-create the environment to speed up the build and improve caching
RUN python3.11 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    python3.11 -m pip install "gradio>=4.29.0" && \
    python3.11 -m pip install fastai

# Set home to the user's home directory and ensure PATH includes user's local bin
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:/usr/local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy the current directory contents into the container at /home/user/app with the new user ownership
COPY --chown=user:user . $HOME/app

# Check if the SSH key is working
RUN git ls-remote git@github.com:github/gitignore.git > /dev/null 2>&1 || exit 1

# Install the flat-bug repo
RUN git clone git@github.com:darsa-group/flat-bug.git && \
    cd flat-bug && \
    git fetch && \
    git checkout dev_experiments && \
    git pull && \
    python3.11 -m pip install -e . && \
    cd ..

# Install the insectnet repo
RUN git clone git@github.com:GuillaumeMougeot/insectnet.git && \
   cd insectnet && \
   pip install -e . && \
   cd ..

# Install the classification dependency
RUN git clone git@github.com:asgersvenning/BiodivX-XPRIZE-Classifier.git && \
    cd BiodivX-XPRIZE-Classifier && \
    git fetch && \
    git checkout main && \
    git pull && \
    cp classify.py .. && \
    cd ..

# Install the localization dependency
RUN git clone git@github.com:asgersvenning/BiodivX-XPRIZE-Localizer.git && \
    cd BiodivX-XPRIZE-Localizer && \
    git fetch && \
    git checkout main && \
    git pull && \
    cp localize.py .. && \
    cd ..

# Install rawpy
RUN cd $HOME && \
    git clone git@github.com:LibRaw/LibRaw.git libraw && \
    git clone git@github.com:LibRaw/LibRaw-cmake.git libraw-cmake && \
    cd libraw && \
    git checkout 0.20.0 && \
    cp -R ../libraw-cmake/* . && \
    cmake . -DCMAKE_INSTALL_PREFIX=$HOME/.local && \
    make && \
    make install && \
    pip install cython && \
    pip install rawpy --user --no-deps && \
    cd ..

# # Install classification dependency
# RUN git clone <REPO> && \
#    cd <REPO> && \
#    git checkout <BRANCH> && \
#    git fetch && \
#    git pull && \
#    pip install -e . <REPO>

# Expose the necessary port for Gradio
EXPOSE 7860
# Sets the GRADIO_SERVER_NAME environment variable to ensure Gradio listens on all network interfaces.
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Command to run your Gradio app
CMD ["python3.11", "app.py"]