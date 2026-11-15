#!/bin/sh
# Builds the engine as a shared library + CLI. Needs only a C++17 compiler.
set -e
cd "$(dirname "$0")"
mkdir -p build
CXX=${CXX:-g++}
FLAGS="-std=c++17 -O3 -march=native -ffast-math -fno-finite-math-only -fopenmp -fvisibility=hidden -Wall -Wno-unused-result -Wno-misleading-indentation"
case "$(uname -s)" in
  Darwin) LIB=build/libterra_fovea.dylib; FLAGS="$(echo $FLAGS | sed 's/-fopenmp//')" ;;
  MINGW*|MSYS*|CYGWIN*) LIB=build/terra_fovea.dll ;;
  *) LIB=build/libterra_fovea.so ;;
esac
$CXX $FLAGS -fPIC -shared core/terra_fovea.cpp -o $LIB
$CXX $FLAGS core/terra_cli.cpp core/terra_fovea.cpp -o build/terra_cli
echo "built $LIB and build/terra_cli"
