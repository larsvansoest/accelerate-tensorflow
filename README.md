# Accelerate Tensorflow
An extension to Accelerate to embed calculations in TensorFlow.

## Running the project locally
This repository was developed on a Unix-based system (Ubuntu with WSL 2) using VSCode Remote (ssh / wsl2). The installation instructions below include steps to run and develop the project.

#### GPU (Nvidia)
In order to run TensorFlow on a Nvidia GPU, follow the guide below to install cuDNN 8.1.1.33 and CUDA 11.2.
- https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html
  > On Ubuntu 20.04, running the script `scripts/ubuntu2004/install-cudnn.sh` should cover the guide.
- https://docs.nvidia.com/cuda/cuda-installation-guide-linux
  > Preferably: download the 11.2.0 CUDA toolkit here: https://developer.nvidia.com/cuda-11.2.0-download-archive
  > Otherwise, for Ubuntu 20.04, running the script `scripts/ubuntu2004/install-cuda.sh` should cover the guide.

### Setting up the project
1. Clone the repository with `git clone git@github.com:larsvansoest/accelerate-tensorflow.git`

2. Run `scripts/setup.sh`.
   > Run with flag `-c` to remove previously installed submodules.
     Run with flag `-g` to switch to gurobi instead of cbcFusion.

3. Make sure `unzip` is installed. Then run `scripts/install.sh`.
  > Run with flag `-g` to switch to TensorFlow for GPU.

### Running the project
Unless using Gurobi, make sure `coinor-cbc, zlib1g-dev, libgl1-mesa-dev freeglut3{,-dev} libsnappy-dev` are installed. To run the project, run the script `scripts/test.sh`.
> In case ghc complains that it cannot find the module 'TensorFlow.Internal.Raw', do the following manually:
    ```sh
    stack exec -- c2hs -C -I -C tensorflow -C -I -C tensorflow/third_party tensorflow/src/TensorFlow/Internal/Raw.chs
    ```

When runing on GPU, you can ignore the could 'NUMA node' info message.

## Developing the project
Make sure the following is installed on the system.
- [VSCode](https://code.visualstudio.com/)
- The [VSCode Haskell extension](https://marketplace.visualstudio.com/items?itemName=haskell.haskell) installed.

1. For starters, install ghcup by following [their installation guide](https://www.haskell.org/ghcup/install/), or by running the command below.
```sh
https://www.haskell.org/ghcup/install/
```
  > ghcup's script will inform about packages that should be installed on the system, ensure those are installed.

2. Run the setup script `scripts/setup.sh` and then the `script/install.sh` script.

2. Navigate to the repository root, and run `stack install`, verify that the code inspection from the VSCode Haskell extension works accordingly.
  > this may not always work right from the start, try running `stack setup` or `stack install` or reloading the window until it does. Otherwise, head to the VSCode Haskell Extension settings and set `Manage HLS` to `Ghcup`. To check the status of compilation, head to `View > OutPut` and select the `Haskell` extension.
  > If VSCode Haskell extension reports an error on ghcide compliation version, run `ghcup set ghc 8.10.7` and reload the window.

# TODO
- Literatuur over het flattenen van een while loop.