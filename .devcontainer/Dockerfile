# accelerate requires nvcc tool, which is included in cuda's devel images.
FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

# required os libraries
ARG LLVM_VERSION=12
ARG CONTAINER_TIMEZONE=Europe/Amsterdam
RUN apt-get update \
    # tzdata is a dependency of the packages, which requires a set timezone
    && ln -snf /usr/share/zoneinfo/${CONTAINER_TIMEZONE} /etc/localtime && echo ${CONTAINER_TIMEZONE} > /etc/timezone \
    # install depedencies to add package sources
    && apt-get install -y gnupg2 curl software-properties-common \
    # add llvm package source
    && curl https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - \
    && echo "deb http://apt.llvm.org/focal/ llvm-toolchain-focal-${LLVM_VERSION} main" >> /etc/apt/sources.list \
    && echo "deb-src http://apt.llvm.org/focal/ llvm-toolchain-focal-${LLVM_VERSION} main" >> /etc/apt/sources.list \
    # add latest stable git source to allow ssh commit signing
    && add-apt-repository ppa:git-core/ppa -y \
    && apt-get update \
    && apt-get install -y \
        # dependencies listed in quickhull practical assignment
        freeglut3 \
        freeglut3-dev \
        pkg-config \
        clang-${LLVM_VERSION} \
        libclang-${LLVM_VERSION}-dev \
        llvm-${LLVM_VERSION}-dev \
        g++ \
        # include latest stable git version
        git \
        # running ghcup failed: make could not be found
        make \
        # running slack failed: ghcup installs stack, but does not install its dependencies
        libgmp10 \
        libgmp-dev \
        # accelerate compilation failed: libcapstone-dev installs missing pkg-config package 'capstone'
        libcapstone-dev \
        # accelerate compilation failed: libfreetype6 installs missing pkg-config package 'freetype2'
        libfreetype6-dev \
        # accelerate compilation failed: libglfw3-dev installs missing pkg-config package'glfw3'
        libglfw3-dev \
        # accelerate compilation failed: libspice-client-gtk-3.0-dev installs missing pkg-config package 'gtk+-3.0'
        libspice-client-gtk-3.0-dev \
        # accelerate compilation failed: accelerate compilation failed when encountering old intel building blocks package
        libtbb-dev

# accelerate build failed: nvvm file could not be found
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/nvvm/lib64:${LD_LIBRARY_PATH}"

# create the vscode user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

# ghcup must be installed by non-root/non-admin user
USER $USERNAME

ARG GHC_VERSION=8.10.7
ARG HLS_VERSION=1.7.0.0
# install ghcup, automatically installs cabal and stack
RUN export BOOTSTRAP_HASKELL_INSTALL_STACK=True \
    export BOOTSTRAP_HASKELL_GHC_VERSION=${GHC_VERSION} \
    && export BOOTSTRAP_HASKELL_NONINTERACTIVE=True \
    && curl --proto '=https' --tlsv1.2 -sSf https://get-ghcup.haskell.org | sh

# add ghcup, cabal and stack to path
ENV PATH="/home/${USERNAME}/.ghcup/bin:${PATH}"

# install Haskell Language Server with ghcup
# RUN ghcup compile hls -v ${HLS_VERSION} --ghc ${GHC_VERSION}

# set c enconding to utf-8
ENV LC_ALL="C.UTF-8"

# set shell to bash to use auto completion (e.g. arrow up for last command)
ENV SHELL="/bin/bash"
