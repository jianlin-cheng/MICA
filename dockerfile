# Dockerfile
FROM nvidia/cuda:11.7.0-cudnn8-devel-ubuntu20.04

# Set environment variables
ARG DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:99
ENV PATH="/opt/conda/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglu1-mesa \
    xvfb \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -ya

# Create and activate MICA environment
RUN conda create -n MICA python=3.8.12 -y

# Install conda packages in MICA environment
RUN conda install -n MICA -y --channel=pytorch --channel=anaconda --channel=conda-forge \
    biopython==1.78 \
    freesasa \
    numpy==1.19.1 \
    pandas==1.1.3 \
    pip==21.2.1 \
    pytorch==1.8.1 \
    scipy==1.5.2 \
    psutil \
    && conda clean -ya

# Install pip packages in MICA environment
RUN /opt/conda/envs/MICA/bin/pip install -U --no-cache-dir --no-deps \
    atom3d==0.2.4 \
    einops \
    mrcfile==1.3.0 \
    natsort \
    networkx==2.6.3 \
    rotary-embedding-torch \
    scikit-image==0.18.2 \
    superpose3d==1.1.1 \
    tqdm==4.61.2 \
    wandb \
    open3d

    
# Create a non-root user (optional but recommended)
RUN useradd -m -s /bin/bash user
USER user
WORKDIR /home/user

# Activate MICA environment by default
RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate MICA" >> ~/.bashrc

# Set the default command
CMD ["/bin/bash", "-l"]