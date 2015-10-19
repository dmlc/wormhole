ifneq ("$(wildcard ../config.mk)","")
	config = ../../config.mk
else
	config = ../../make/config.mk
endif

include $(config)
include ../../repo/dmlc-core/make/dmlc.mk

BIN = linear.dmlc

# common build script for programs

export LDFLAGS= -L../../deps/lib -pthread -lm -lrt $(DMLC_LDFLAGS)
export CFLAGS = -Wall  -msse2  -Wno-unknown-pragmas -fPIC -I../../repo/rabit/include  -I../../repo/dmlc-core/include -std=c++11 -fopenmp

linear.dmlc : linear.cc ../../repo/dmlc-core/libdmlc.a ../../repo/rabit/lib/librabit.a

all: $(BIN) 

$(BIN) : 
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c %.cc %.a,  $^) $(LDFLAGS)

clean:
	$(RM) $(OBJ) $(BIN)  *~ ../src/*~

