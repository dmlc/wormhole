ifneq ("$(wildcard ../config.mk)","")
	config = ../../config.mk
else
	config = ../../make/config.mk
endif

export CC = /data1/home/wangchao/gcc-4.8.2/bin/gcc-4.8
export CXX = /data1/home/wangchao/gcc-4.8.2/bin/g++-4.8

include $(config)
include ../../repo/dmlc-core/make/dmlc.mk

BIN = fm.dmlc

# common build script for programs

export LDFLAGS= -L../../lib -pthread -lm -lrt $(DMLC_LDFLAGS)
export CFLAGS = -Wall  -msse2  -Wno-unknown-pragmas -fPIC -I../../repo/rabit/include  -I../../repo/dmlc-core/include -std=c++11 -fopenmp

ifdef DEPS_PATH
CFLAGS+= -I$(DEPS_PATH)/include
LDFLAGS+= -L$(DEPS_PATH)/lib
endif

fm.dmlc : fm.cc ../../repo/dmlc-core/libdmlc.a ../../repo/rabit/lib/librabit.a

all: $(BIN) 

$(BIN) : 
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c %.cc %.a,  $^) $(LDFLAGS)

clean:
	$(RM) $(OBJ) $(BIN)  *~ ../src/*~

