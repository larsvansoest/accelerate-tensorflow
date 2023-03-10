# Accelerate Tensorflow
An extension to Accelerate to embed calculations in TensorFlow.

## Running the project locally
This repository was developed on a Unix-based system (Ubuntu with WSL 2) using VSCode Remote (ssh / wsl2). The installation instructions below include steps to run and develop the project.

### Packages
Make sure the following packages are installed:
- `apt install coinor-cbc`

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
  > In case ghc complains that it cannot find the module 'TensorFlow.Internal.Raw', do the following manually:
    ```sh
    stack exec -- c2hs -C -I -C tensorflow -C -I -C tensorflow/third_party tensorflow/src/TensorFlow/Internal/Raw.chs
    ```

### Compiling the source
1. Verify the accelerate installation by running `stack build` inside the `extra-deps/accelerate` folder.
2. Verify the project installation by running `stack test` in the root folder of this repository.

### Running the project
To run the project, run the script `scripts/test.sh`.

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