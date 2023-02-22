# Template for Accelerate Projects
This template contains the basis for an [Accelerate](https://github.com/AccelerateHS/accelerate) project.

This template will use the interpreter for Accelerate. For CPU and GPU, respectively use this repository's [cpu](https://github.com/larsvansoest/template-accelerate/tree/cpu) and [gpu](https://github.com/larsvansoest/template-accelerate/tree/gpu) branches.

## Setting up the dev environment (Nvidia GPU + Windows 11 + WSL2)

The repository contains [dev container](https://code.visualstudio.com/docs/remote/containers) files to setup a vscode dev container to run [Accelerate](https://github.com/AccelerateHS/accelerate). 

The environment provides the following features:
- An Ubuntu container with all the required packages installed to run accelerate.
- [Cuda](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) support to run [Accelerate](https://github.com/AccelerateHS/accelerate) on the host machine's nvidia gpu from the dev container.
- [VSCode Haskell extension](https://marketplace.visualstudio.com/items?itemName=haskell.haskell), with code highlights, hlint and mouse tooltip type inspection.
- [GitHub SSH Authentication](https://docs.github.com/en/authentication/connecting-to-github-with-ssh) and [GitHub SSH Signing](https://docs.github.com/en/authentication/managing-commit-signature-verification/signing-commits)

### Requirements
- [Windows 11](https://www.microsoft.com/en-us/windows/windows-11?r=1)
- [WSL2 with Ubuntu distro](https://docs.microsoft.com/en-us/windows/wsl/install)
- [VSCode](https://code.visualstudio.com/)
    - With the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) installed.
- [Docker with WSL2 backend](https://www.docker.com/get-started/)

### Installation

Aside from the GitHub SSH setup steps below, the dev container works right out of the box.

Simply open this repository in VSCode with the remote containers extension installed, hit `ctrl + shift + p` and select `Remote-Containers: Rebuild and Reopen in Container`.

After around 10 to 15 minutes of building, the container should be ready to use.

> Note: after (re)building, when first clicking a Haskell file, the Haskell plugin has to compile the package. 
    In vscode's notifications (footer bar), there will be an indication that it is still loading. This will resolve in around 10-15 minutes and only has to run after the first time of building the container.

    To check the status of compilation, head to `View > OutPut` and select the `Haskell` extension.

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
Run the command `nvidia-smi` in the dev container, the result should somewhat similar to the following.
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

#### HLS Server crashed
If the Haskell plugin returns `The HLS server crashed 5 times in the past 3 minutes`, open up a terminal and type `stack build`. The plugin should run correctly after reloading the window.

#### File Error Highlight: ensure that ghcide is compiled with the same GHC
If a file is highlighted with the error `Please ensure that ghcide is compiled with the same GHC`, running `stack build` then closing and opening the file may resolve the issue. Otherwise, reload the window.

#### Git Commit Error: unable to start editor 'editor'
When merging branches locally, if git throws `unable to start editor 'editor'`, it can not find a suitable editor. To resolve this issue, ensure the git config contains `core.editor = code`. For an example of a correct git config, see [Setting up GitHub SSH Authentication & SSH Signing](#setting-up-github-ssh-authentication--ssh-signing).

# Benodigdheden TF Pipeline
- MakesILP (Fusion) doet David
- SimplifyOperation is makkelijk (check voor TId)
- SLVOperation: Return altijd Nothing
- PrettyOp: Al geimplementeerd
- KernelOperation (TensorKernel)
    
    `clusterOperations` en `ClusterOperations` uit `Data.Array.Accelerate.Eval`
    `compileKernel` uit een type class

    ```hs
    compileKernel :: Env AccessGroundR env -> Cluster TensorOp args -> Args env args -> TensorKernel env
    compileKernel env cluster clusterArgs =
    case clusterOperations cluster clusterArgs of
        ClusterOperations _ (LeftHandSideWildcard _) [ApplyOperation operation args] -> compileOperation env operation args
        _ -> internalError "Expected a cluster with one operation"

    compileOperation :: Env AccessGroundR env -> TensorOp args -> Args env args -> TensorKernel env
    compileOperation = undefined
    ```
    - `Array sh a` becomes `ShapeR sh, TypeR a, ExpVars env sh, BaseVar env (Buffers a)`.
    - If a is a scalar type, then `ScalarType a` instead of TypeR and Buffer instead of Buffers.
    - The type evidence (TypeR or ScalarType) might already be present somewhere else (in PrimFun for instance), then it can be omitted.
    ```hs
    data TensorKernel env where
    TensorPrimFun :: ShapeR sh -> PrimFun (a -> b) -> ExpVars env sh -> BaseVars env (Buffers a) -> BaseVars env (Buffers b) -> TensorKernel env
    TensorId :: ShapeR sh -> ScalarType a -> BaseVar env (Buffer a) -> BaseVar env (Buffer b) -> TensorKernel env
    TensorConst :: ShapeR sh -> ScalarType a -> ExpVars env sh -> a -> BaseVar env (Buffer a) -> TensorKernel env
    ```
- PrettyKernel
- IsSchedule al geimplementeerd (@UniformScheduleFun)
- PrettySchedule is ook al geimplementeerd
- IsKernel (zie compileOperation hierboven, etc., )
- Operation.ShrinkArg (David)
- Partitioning.BackendClusterArg (David)

### Priorities
- TF Pipeline
- Mkgenerate
- Mkpermute
