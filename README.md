# Accelerate Tensorflow
An extension to Accelerate to embed calculations in TensorFlow.

## Setting up the dev environment
This repository was developed on a Unix-based system (Ubuntu with WSL 2) using VSCode Remote (ssh / wsl2). The installation instructions below include steps to run and develop the project.

### Setting up the project
1. Clone the repository with `git clone git@github.com:larsvansoest/accelerate-tensorflow.git`

> Instead of following steps 2 and 3 below manually, run `scripts/setup.sh`.

2. Inside the repository folder, clone the gitsubmodules with the following steps:
  - `git submodule update --init --recursive --remote -- extra-deps/accelerate`, make sure `--remote` is included to fetch the correct branch.
  - `git submodule update --init --recursive -- extra-deps/tensorflow-haskell`, make sure `--remote` is *not* included to ensure the tensorflow verion is pinned to 2.10.1.

3. Modify `extra-include-dirs` and `extra-lib-dirs` in `extra-deps/tensorflow-haskell/stack.yaml` as follows:
  ```yaml
  extra-lib-dirs:
      - /usr/local/lib
      - ../tf2101/lib
  extra-include-dirs:
      - /usr/local/include
      - tensorflow
      - tensorflow/third_party
  ```

### Installing the required packages

> Instead of following steps 1-4 below manually, run `scripts/install.sh`.

1. Install protoc as follows
```sh
curl -O -L https://github.com/google/protobuf/releases/download/v3.13.0/protoc-3.13.0-linux-x86_64.zip
unzip -d ~ protoc-3.13.0-linux-x86_64.zip bin/protoc
chmod 755 ~/bin/protoc
rm protoc-3.13.0-linux-x86_64.zip
```

2. Add protoc to PATH as by modifying `~/.bashrc` by adding the following line.
```txt
export PATH="${PATH}:/home/<username>/bin/protoc/"
```

3. Install tensorflow 2.10.1 as follows
```sh
mkdir -p extra-deps/tf2101
curl -O https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-2.10.1.tar.gz
tar zxf libtensorflow-cpu-linux-x86_64-2.10.1.tar.gz -C extra-deps/tf2101
rm libtensorflow-cpu-linux-x86_64-2.10.1.tar.gz
```

4. Run `env LD_LIBRARY_PATH="<path-to-project>/extra-deps/tf2101/lib:$LD_LIRBARY_PATH" stack test`, inside the `extra-deps/tensorflow-haskell` folder.
  - In case ghc complains that it cannot find the module 'TensorFlow.Internal.Raw', do the following manually:
    ```sh
    stack exec -- c2hs -C -I -C tensorflow -C -I -C tensorflow/third_party tensorflow/src/TensorFlow/Internal/Raw.chs
    ```
  - Without step 4, 

### Compiling the source
1. Verify the accelerate installation by running `stack build` inside the `extra-deps/accelerate` folder.
2. Verify the project installation by running `stack test` in the root folder of this repository.

### Running the project
Make sure the following is installed on the system.
- [Gurobi](https://www.gurobi.com/documentation/10.0/quickstart_linux/index.html)
  > Gurobi requires a license. If there are any issues installing or using Gerobi, consider switching to cbc by following the instructions below.

To run the project, run the script `scripts/test.sh`.

#### Gurobi alternative
In order to not use Gerobi, modify the following in `/extra-deps/accelerate/src/Data/Array/Accelerate/Trafo/NewNewFusion.hs`
- Replace the existing `Data.Array.Accelerate.Trafo.Partitioning.ILP `s import with `import Data.Array.Accelerate.Trafo.Partitioning.ILP (gurobiFusion, gurobiFusionF)`.
- Replace any occurence of `gurobiFusion` with `cbcFusion`, and `gurobiFusionF` with `cbcFusionF`.
> Note: accelerate's current implemenation requires an alternative to MIP, in order to make cbc work, please replace the MIP dependency with https://github.com/dpvanbalen/haskell-MIP/commit/3a65b4479bf4dd191077b567b381b79ca5b4f1fa

### Developing the project
Make sure the following is installed on the system.
- [VSCode](https://code.visualstudio.com/)
- The [VSCode Haskell extension](https://marketplace.visualstudio.com/items?itemName=haskell.haskell) installed.

1. For starters, install ghcup by following [their installation guide](https://www.haskell.org/ghcup/install/), or by running the command below.
```sh
https://www.haskell.org/ghcup/install/
```
  > ghcup's script will inform about packages that should be installed on the system, ensure those are installed.

2. In `extra-deps/accelerate`, create a new file named `stack.yaml` and copy the contents of `stack-8.10.yaml` in there. Then run `stack install` in `extra-deps/accelerate`. When finished hit `ctrl + shift + p` and select `> Reload Window`. The Haskell extension will prompt to install the required stack, click yes on every option. After a while, the code inspection should work for the accelerate source.

3. Navigate to the repository root, and run `stack install`, verify that the code inspection from the VSCode Haskell extension works accordingly.
  > this may not always work right from the start, try running `stack setup` or `stack install` or reloading the window until it does. Otherwise, head to the VSCode Haskell Extension settings and set `Manage HLS` to `Ghcup`. To check the status of compilation, head to `View > OutPut` and select the `Haskell` extension.

### Trouble Shooting
When in trouble, perhaps the instructions below might help.

#### Verify if GPU is correctly working
Run the command `nvidia-smi`, the result should somewhat similar to the following.
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 515.65.01    Driver Version: 516.94       CUDA Version: 11.7     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  On   | 00000000:07:00.0  On |                  N/A |
|  0%   30C    P8    28W / 370W |   1512MiB / 10240MiB |      6%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

#### Can not find 'TensorFlow.Internal.Raw'
In case ghc complains that it cannot find the module 'TensorFlow.Internal.Raw', do the following manually:
```sh
stack exec -- c2hs -C -I -C tensorflow -C -I -C tensorflow/third_party tensorflow/src/TensorFlow/Internal/Raw.chs
```

#### Ghcide was built with version x, but using version y
If VSCode Haskell extension reports an error on ghcide compliation version, run `ghcup set ghc 8.10.7` and reload the window.