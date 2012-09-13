# C_INCLUDE_PATH = /usr/include/python2.6:/System/Library/Frameworks/vecLib.framework/Versions/Current/Headers
C_INCLUDE_PATH = /usr/include/python2.6:/opt/local/include

all: cython SMO.c
	@export C_INCLUDE_PATH=$(C_INCLUDE_PATH)
	gcc -framework Python -bundle -bundle_loader `which python` SMO.c -o SMO.so -I/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages/numpy/core/include -L/opt/local/lib

cython: SMO.pyx
	cython SMO.pyx

clean:
	rm SMO.c SMO.so

