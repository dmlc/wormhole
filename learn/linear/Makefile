# global config
ifneq ("$(wildcard ../../config.mk)","")
	config = ../../config.mk
else
	config = ../../make/config.mk
endif

include $(config)

DEPS_PATH=/usr/local/

CORE_PATH=../../dmlc-core
include $(CORE_PATH)/make/dmlc.mk
PS_PATH=../../repo/ps-lite
include $(PS_PATH)/make/ps.mk

INC=-I./ -I$(PS_PATH)/src -I$(CORE_PATH)/include -I$(CORE_PATH)/src
CFLAGS  = -O3 -ggdb -Wall -std=c++11 $(DMLC_CFLAGS) $(PS_CFLAGS) $(INC)
LDFLAGS = $(CORE_PATH)/libdmlc.a $(DMLC_LDFLAGS) $(PS_PATH)/build/libps.a $(PS_LDFLAGS) #-pthread -lm

all: bin test # build/linear test

clean:
	rm -rf build/*

bin: build/ftrl build/async_sgd

build/%.o: %.cc
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -MM -MT build/$*.o $< >build/$*.d
	$(CXX) $(CFLAGS) -c $< -o $@

%.pb.cc %.pb.h : %.proto
	${DEPS_PATH}/bin/protoc --cpp_out=. --proto_path=. $<

test: build/localizer_test build/spmv_test build/minibatch_iter_test \
build/arg_parser\

build/async_sgd: build/sgd/async_sgd.o \
build/proto/config.pb.o build/proto/sys.pb.o
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

build/ftrl: build/sgd/ftrl.o build/proto/config.pb.o build/proto/sys.pb.o
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

build/%: build/test/%.o
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

-include build/*/*.d
-include build/*/*/*.d
