# This script runs stack test and refers to the correct tf2101 library.

SCRIPT=$(realpath "$0")
SCRIPT_PATH=$(dirname "$SCRIPT")

PROJECT_DIR=$(dirname -- "$0")/..

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$SCRIPT_PATH/../extra-deps/tf2101/lib"
export TF_CPP_MIN_LOG_LEVEL=2

cd $PROJECT_DIR
mkdir -p "test-logs"
cur_date=$(date +%s)
stack test > $SCRIPT_PATH/../test-logs/test-${cur_date}.log