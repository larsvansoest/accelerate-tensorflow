# This script sets up the project files.
# Run with -c to remove previously installed submodules.
# Run with -g to switch to gurobi instead of cbcFusion.

CLEAN=false
GUROBI=false
while getopts "cg" flag; do
  case $flag in
    c) CLEAN=true ;;
    g) GUROBI=true
  esac
done

if [[ $CLEAN == true ]]
then
  echo "[Setup] Cleaning submodules files."
  git submodule deinit -f --all
fi

SCRIPT_DIR=$(dirname -- "$0")

echo "[Setup] Cloning submodules."

git submodule update -f --init --recursive --remote -- extra-deps/accelerate
git submodule update -f --init --recursive --remote -- extra-deps/haskell-MIP
git submodule update -f --init --recursive -- extra-deps/tensorflow-haskell

echo "[Setup] Copying configuration files into tensorflow-haskell submodule"

cp --verbose -r $SCRIPT_DIR/setup-files/tensorflow-haskell/* $SCRIPT_DIR/../extra-deps/tensorflow-haskell

echo "[Setup] Copying configuration files into accelerate submodule"
cp --verbose $SCRIPT_DIR/../extra-deps/accelerate/stack-8.10.yaml $SCRIPT_DIR/../extra-deps/accelerate/stack.yaml
cp --verbose $SCRIPT_DIR/setup-files/accelerate/accelerate.cabal $SCRIPT_DIR/../extra-deps/accelerate/accelerate.cabal

if [[ $GUROBI == false ]]
then
  echo "[Setup] Copying CBC fusion instructions to accelerate submodule"
  cp --verbose $SCRIPT_DIR/setup-files/accelerate/fusion/cbc/NewNewFusion.hs $SCRIPT_DIR/../extra-deps/accelerate/src/Data/Array/Accelerate/Trafo/NewNewFusion.hs
else
  echo "[Setup] Copying Gurobi fusion instructions to accelerate submodule"
  cp --verbose $SCRIPT_DIR/setup-files/accelerate/fusion/gurobi/NewNewFusion.hs $SCRIPT_DIR/../extra-deps/accelerate/src/Data/Array/Accelerate/Trafo/NewNewFusion.hs
fi