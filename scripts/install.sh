# This script installs the required libraries to run this project.
# Note: to install tensorflow for gpu, run with `install.sh -g`.

SCRIPT_DIR=$(dirname -- "$0")
TF2021_DIR="$SCRIPT_DIR/../extra-deps/tf2101"
PROTOC_DIR=~/bin/protoc

echo "[Install] Removing previously installed files at $TF2021_DIR and $PROTOC_DIR."
rm -f -r $TF2021_DIR
rm -f -r $PROTOC_DIR

echo "[Install] Installing protoc at $PROTOC_DIR"

curl -O -L https://github.com/google/protobuf/releases/download/v3.13.0/protoc-3.13.0-linux-x86_64.zip
unzip -d ~ protoc-3.13.0-linux-x86_64.zip bin/protoc
chmod 755 $PROTOC_DIR
rm protoc-3.13.0-linux-x86_64.zip

echo "[Install] Protoc installed, please add $PROTOC_DIR to PATH"

mkdir -p $TF2021_DIR

GPU=false
while getopts "g" flag; do
  case $flag in
    g) GPU=true
  esac
done

if [[ $GPU == false ]]
then
  echo "[Intall] Installing Tensorflow 2.10.1 (CPU) at $TF2021_DIR"
  curl -O https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-2.10.1.tar.gz
  tar zxf libtensorflow-cpu-linux-x86_64-2.10.1.tar.gz -C $TF2021_DIR
  rm libtensorflow-cpu-linux-x86_64-2.10.1.tar.gz
else
  echo "[Intall] Installing Tensorflow 2.10.1 (GPU) at $TF2021_DIR"
  curl -O https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-2.10.1.tar.gz
  tar zxf libtensorflow-gpu-linux-x86_64-2.10.1.tar.gz -C $TF2021_DIR
  rm libtensorflow-gpu-linux-x86_64-2.10.1.tar.gz
fi


echo "[Install] Tensorflow installed"