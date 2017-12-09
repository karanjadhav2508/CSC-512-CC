#!/bin/bash

BUILD_DIR="$HOME/llvm/build-release"
TOOLS_EXTRA="$HOME/llvm/llvm/tools/clang/tools/extra"
ARGS="$HOME/final_package/lonestargpu-2.0/apps/mst_dp_modular/main.cu -- --cuda-host-only --cuda-gpu-arch=sm_35 -I /usr/local/cuda/include -I $HOME/final_package/lonestargpu-2.0/include -I $HOME/final_package/lonestargpu-2.0/cub-1.7.4 -w"

#copy tools to clang extra sub-folder
cp -r transformers/t1 "$TOOLS_EXTRA/"
cp -r transformers/t2 "$TOOLS_EXTRA/"
cp -r transformers/t3 "$TOOLS_EXTRA/"
cp -r transformers/t4 "$TOOLS_EXTRA/"

#add tool details to common CMakeList file
echo 'add_subdirectory(t1)' >> "$TOOLS_EXTRA/CMakeLists.txt"
echo 'add_subdirectory(t2)' >> "$TOOLS_EXTRA/CMakeLists.txt"
echo 'add_subdirectory(t3)' >> "$TOOLS_EXTRA/CMakeLists.txt"
echo 'add_subdirectory(t4)' >> "$TOOLS_EXTRA/CMakeLists.txt"

#compile tools
cd $BUILD_DIR
ninja

#add symlink for cub in lonestar
ln -s "$HOME/final_package/cub-1.7.4" "$HOME/final_package/lonestargpu-2.0/cub-1.7.4"

#run sample source file against each binary
$BUILD_DIR/bin/t1 $ARGS
$BUILD_DIR/bin/t2 $ARGS
$BUILD_DIR/bin/t3 $ARGS
$BUILD_DIR/bin/t4 $ARGS
