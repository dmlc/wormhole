ifndef config
	ifneq ("$(wildcard ../../config.mk)","")
		config = ../../config.mk
	else
		config = ../../make/config.mk
	endif
endif

include $(config)
include ../../dmlc-core/make/dmlc.mk

export LDFLAGS= -L../../lib -pthread -lm -lrt $(DMLC_LDFLAGS)
export CFLAGS = -Wall  -msse2  -Wno-unknown-pragmas -fPIC -I../../repo/rabit/include  -I../../dmlc-core/include -std=c++11

BIN = kmeans.dmlc
OBJ =

kmeans.dmlc : kmeans.cc ../../dmlc-core/libdmlc.a ../../repo/rabit/lib/librabit.a

all: $(BIN) 

$(BIN) : 
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c %.cc %.a,  $^) $(LDFLAGS)

clean:
	$(RM) $(OBJ) $(BIN)  *~ ../src/*~
