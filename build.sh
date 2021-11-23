#!/bin/bash
if [ ! -d build ]
then
	mkdir build
fi

if [ ! -d /opt/custom ]
then
	mkdir -p /opt/custom/ptx
	mkdir -p /opt/custom/bin/test
fi

GPUSMS="MAX_SM=6"
GPUCC="GPU_CC=61"

cmake -DCMAKE_BUILD_TYPE=DEBUG -DCppUTILITY=ON -DBUILD_TEST=ON -D$GPUSMS -D$GPUCC -H. -Bbuild
cmake --build build -- -j 12
cmake -P build/cmake_install.cmake > /dev/null
