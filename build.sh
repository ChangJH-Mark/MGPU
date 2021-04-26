#!/bin/bash
if [ ! -d build ]
then
	mkdir build
fi

if [ ! -d /opt/custom ]
then
	mkdir -p /opt/custom/ptx
fi

cmake -DCMAKE_BUILD_TYPE=DEBUG -DCppUTILITY=ON -H. -Bbuild
cmake --build build -- -j 12 
