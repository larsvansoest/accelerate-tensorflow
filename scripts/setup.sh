# This script sets up the project files.

SCRIPT_DIR=$(dirname -- "$0")

echo "[Setup] Cloning submodules."

git submodule update --init --recursive --remote -- extra-deps/accelerate
git submodule update --init --recursive -- extra-deps/tensorflow-haskell

echo "[Setup] Copying tensorflow-haskell configuration files into submodule source"

cp -r $SCRIPT_DIR/setup-files/tensorflow-haskell/* $SCRIPT_DIR/../extra-deps/tensorflow-haskell
