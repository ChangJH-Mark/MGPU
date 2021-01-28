#!/bin/bash
if [ ! -d build ]
then
	mkdir build
fi

cmake -H. -Bbuild
cmake --build build -- -j 12 
