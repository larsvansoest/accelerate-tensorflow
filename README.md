# Template for Accelerate Projects
This template contains the basis for an [Accelerate](https://github.com/AccelerateHS/accelerate) project.

This template will use the interpreter for Accelerate. For CPU and GPU, respectively use this repository's [cpu](https://github.com/larsvansoest/template-accelerate/tree/cpu) and [gpu](https://github.com/larsvansoest/template-accelerate/tree/gpu) branches.

## Setting up the dev environment
This repository was developed on a Unix-based system (Ubuntu with WSL 2) using VSCode Remote (ssh / wsl2). The installation instructions below include steps to run and develop the project.

### Running the project
Make sure the following is installed on the system.
- [Gurobi](https://www.gurobi.com/documentation/10.0/quickstart_linux/index.html)
  > Gurobi requires a license. If there are any issues installing or using Gerobi, consider switching to cbc by following the instructions below.

1. Clone this repository, and run `git submodule update --init --recursive --remote` to load the required submodules. This copies the source of the accelerate project inside of the `extra-deps/accelerate` folder. This allows for easy code inspections and to modify the source when necessary.
  > It is important that `--remote` is added to the submodule command, such that the submodule branch is correctly checked out.

2. Run `stack test` in the project root.

#### Gurobi alternative
In order to not use Gerobi, modify the following in `/extra-deps/accelerate/src/Data/Array/Accelerate/Trafo/NewNewFusion.hs`
- Replace the existing `Data.Array.Accelerate.Trafo.Partitioning.ILP `s import with `import Data.Array.Accelerate.Trafo.Partitioning.ILP (gurobiFusion, gurobiFusionF)`.
- Replace any occurence of `gurobiFusion` with `cbcFusion`, and `gurobiFusionF` with `cbcFusionF`.

### Developing on the project
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

#### Setting up GitHub SSH Authentication & SSH Signing
The dev container supports GitHub SSH Authentication and GitHub SSH Signing.

In order to enable this, perform the steps below.

Setup WSL2 with Ubuntu as follows:
1. Create a `~/.ssh` folder with the following files.
    - Add github ssh private and public key to `~/.ssh/<KEYNAME>`, `~/.ssh/<KEYNAME>.pub`
    - Add config file to `~/.ssh/config` with the following contents
        ```
        Host github.com
            user git
            IdentityFile ~/.ssh/<KEYNAME>
        ```
2. Install packages `keychain`.
3. Create `~/.bash_profile` and add the following line: 
    ```
    eval `keychain --eval --agents ssh github_30112021`. 
    ```
4. Setup `~/.gitconfig` as follows
    ```
    [user]
        email = <EMAIL>
        name = <NAME>
        signingkey = <CONTENT OF PUBLIC KEY (e.g. ssh-ed25519 AAAAC3(...) user@example.com)>
    [commit]
        gpgsign = true
    [gpg]
        format = ssh
    [tag]
        gpgsign = true
    [core]
	    editor = code
    ```

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

#### Git Commit Error: unable to start editor 'editor'
When merging branches locally, if git throws `unable to start editor 'editor'`, it can not find a suitable editor. To resolve this issue, ensure the git config contains `core.editor = code`. For an example of a correct git config, see [Setting up GitHub SSH Authentication & SSH Signing](#setting-up-github-ssh-authentication--ssh-signing).
s