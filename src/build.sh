#!/bin/sh
set -e

mkdir -p bin/

# Helper variables
args=""
optimizations="-O3 -DNDEBUG"

args="${args} ${optimizations} --std=c++11 -lineinfo --expt-extended-lambda"
args="${args} -gencode arch=compute_50,code=sm_50 -gencode arch=compute_61,code=sm_61"
args="${args} -maxrregcount=64 -Idynasoar -I. -Idynasoar/lib/cub"

nvcc --shared -Xcudafe "--diag_suppress=1427" -Xcompiler -fPIC ${args} "$@"
