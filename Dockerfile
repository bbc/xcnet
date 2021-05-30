FROM nvidia/cuda:10.2-base-ubuntu18.04

# ENV http_proxy="http://proxy.example.com:port"
# ENV https_proxy="http://proxy.example.com:port"

RUN apt-get update && apt-get install -y \
    curl \
    pv \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    ffmpeg libsm6 libxext6 \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir /app
WORKDIR /app

RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

ENV HOME=/home/user
RUN chmod 777 /home/user

ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py38_4.8.2-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda install -y python==3.8.1 \
 && conda clean -ya

RUN conda install -y -c pytorch \
    cudatoolkit=10.2 \
    "pytorch=1.5.0=py3.8_cuda10.2.89_cudnn7.6.5_0" \
    "torchvision=0.6.0=py38_cu102" \
 && conda clean -ya

RUN pip install tensorboard tqdm scikit-learn==0.23.2 scikit-image mkl jupyter notebook
RUN  conda install -c pytorch faiss-gpu
RUN conda install -c conda-forge opencv \
 && conda clean -ya