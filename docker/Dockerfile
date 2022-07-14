ARG CUDA_DOCKER_VERSION=11.2.2-devel-ubuntu20.04
FROM nvidia/cuda:${CUDA_DOCKER_VERSION}

# MAINTAINER lutianen 

# Arguments for the build. CUDA_DOCKER_VERSION needs to be repeated because
# the first usage only applies to the FROM tag.
# TensorFlow version is tightly coupled to CUDA and cuDNN so it should be selected carefully
ARG CUDA_DOCKER_VERSION=11.2.2-devel-ubuntu18.04
ARG PYTORCH_VERSION=1.8.1+cu111
ARG PYTORCH_VERSION_WEB_1=cu111
ARG PYTORCH_VERSION_WEB_2=1.8.1%2Bcu111
ARG TORCHVISION_VERSION=0.9.1
ARG CUDNN_VERSION=8.1.1.33-1+cuda11.2
ARG NCCL_VERSION=2.8.4-1+cuda11.2

# python version
ARG PYTHON_VERSION=3.8
ARG PYTHON_VERSION_SIMPLE=38
ARG PIP_DOUBAN=https://pypi.douban.com/simple

# FOR Install mpi4py.
ARG MPI_KIND=OpenMPI


# to avoid interaction with apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]


# Prepare to install specific g++ versions
RUN apt-get update -qq && apt-get install -y --no-install-recommends software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test


# Install essential packages.
RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential \
        cmake \
        g++-7 \
        git \
        curl \
        vim \
        wget \
        ca-certificates \
        libcudnn8=${CUDNN_VERSION} \
        net-tools \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
RUN apt-get update
RUN apt install libnccl2=2.8.4-1+cuda11.2 libnccl-dev=2.8.4-1+cuda11.2 && apt-get clean && rm -rf /var/lib/apt/lists/*


RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        libjpeg-dev \
        # libpng-dev \

        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-distutils \
        
        librdmacm1 \
        libibverbs1 \
        ibverbs-providers \
        openjdk-8-jdk-headless \
        openssh-client \
        openssh-server \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


# Install Python.
RUN ln -s -f /usr/bin/python${PYTHON_VERSION} /usr/bin/python
RUN ln -s -f /usr/bin/python${PYTHON_VERSION} /usr/bin/python${PYTHON_VERSION/%.*/}
RUN wget --progress=dot:mega https://bootstrap.pypa.io/get-pip.py && python get-pip.py && rm get-pip.py


# pinning pip to 21.0.0 as 22.0.0 cannot fetch pytorch packages from html linl
# https://github.com/pytorch/pytorch/issues/72045
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py && \
    pip install --no-cache-dir -U --force pip~=21.0.0 -i ${PIP_DOUBAN}


# Add launch helper scripts
# ...
# Install Spark stand-alone cluster.
# ...
# Install PySpark.
# ...


# Install Ray.
# ....


# Install Open MPI
RUN wget --progress=dot:mega -O /tmp/openmpi-3.0.0-bin.tar.gz https://github.com/horovod/horovod/files/1596799/openmpi-3.0.0-bin.tar.gz && \
    cd /usr/local && \
    tar -zxf /tmp/openmpi-3.0.0-bin.tar.gz && \
    ldconfig && \
    mpirun --version



# Allow OpenSSH to talk to containers without asking for confirmation
RUN mkdir -p /var/run/sshd
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config



# Set default NCCL parameters
RUN echo NCCL_DEBUG=INFO >> /etc/nccl.conf


# Install python3.8-dev
RUN apt-get update 
RUN apt-get -y upgrade
RUN apt-get install -y aptitude
RUN aptitude install -y python${PYTHON_VERSION}-dev && apt-get clean && rm -rf /var/lib/apt/lists/*


# Install mpi4py.
# This requires SETUPTOOLS_USE_DISTUTILS=stdlib as with setuptools>=60.1.0 installing mpi4py broke
# https://github.com/mpi4py/mpi4py/issues/157#issuecomment-1001022274
RUN if [[ ${MPI_KIND} != "None" ]]; then \
        SETUPTOOLS_USE_DISTUTILS=stdlib pip install --no-cache-dir mpi4py -i ${PIP_DOUBAN}; \
    fi


# Install PyTorch (releases).
RUN wget https://download.pytorch.org/whl/${PYTORCH_VERSION_WEB_1}/torch-${PYTORCH_VERSION_WEB_2}-cp${PYTHON_VERSION_SIMPLE}-cp${PYTHON_VERSION_SIMPLE}-linux_x86_64.whl


# Install essential packages
RUN apt-get update && apt install -y libgl1-mesa-glx
RUN pip install --no-cache-dir numpy opencv-python matplotlib filelock scipy -i ${PIP_DOUBAN}

RUN pip install --no-cache-dir torch-${PYTORCH_VERSION}-cp${PYTHON_VERSION_SIMPLE}-cp${PYTHON_VERSION_SIMPLE}-linux_x86_64.whl
RUN pip install --no-cache-dir torchvision==0.9.1 -i ${PIP_DOUBAN}

# RUN pip install pytorch_lightning



# RUN wget https://cn.download.nvidia.com/XFree86/Linux-x86_64/510.54/NVIDIA-Linux-x86_64-510.54.run
# RUN chmod a+x NVIDIA-Linux-x86_64-510.54.run
# RUN sh ./NVIDIA-Linux-x86_64-510.54.run -no-x-check -no-nouveau-check -no-opengl-files

# Install essential packages
RUN apt-get update && apt install -y libgl1-mesa-glx
RUN pip install --no-cache-dir numpy opencv-python matplotlib filelock scipy hdf5storage -i ${PIP_DOUBAN}


# Show the effective python package version to easily spot version differences
RUN pip freeze | sort



# test
RUN echo "----------------------\n" && cd /usr/include/ && ls | grep nccl && cd /usr/lib/x86_64-linux-gnu/ && ls | grep nccl && echo "\n-----------------------"


# Install Horovod
ARG HOROVOD_VERSION=0.21.0
RUN MAKEFLAGS="-j6" HOROVOD_WITH_MPI=1 HOROVOD_NCCL_INCLUDE=/usr/include HOROVOD_NCCL_LIB=/usr/lib/x86_64-linux-gnu HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITHOUT_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 pip install --no-cache-dir horovod==${HOROVOD_VERSION} -i ${PIP_DOUBAN}

# Check all frameworks are working correctly. Use CUDA stubs to ensure CUDA libs can be found correctly
# when running on CPU machine
RUN ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
    python -c "import horovod.torch as hvd; hvd.init()" && \
    ldconfig


# workdir
WORKDIR /home/


# CMD
CMD ["/bin/bash"]



