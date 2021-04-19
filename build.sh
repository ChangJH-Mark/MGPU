#!/bin/bash
if [ ! -d build ]
then
	mkdir build
fi

if [ ! -d /opt/custom ]
then
	mkdir /opt/custom
fi

cmake -DCppUTILITY=ON -H. -Bbuild
cmake --build build -- -j 12 
