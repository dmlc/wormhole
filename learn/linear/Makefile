ifneq ("$(wildcard ../../config.mk)","")
	config = ../../config.mk
else
	config = ../../make/config.mk
endif
include $(config)

CORE_PATH=../../dmlc-core
include $(CORE_PATH)/make/dmlc.mk

PS_PATH=../../repo/ps-lite
include $(PS_PATH)/make/ps.mk
ifndef DEPS_PATH
DEPS_PATH = $(PS_PATH)/deps
endif

INC=-I./ -I$(PS_PATH)/src -I$(CORE_PATH)/include -I$(CORE_PATH)/src -I$(DEPS_PATH)/include
CFLAGS  = -O3 -ggdb -Wall -std=c++11 $(DMLC_CFLAGS) $(PS_CFLAGS) $(INC) $(EXTRA_CFLAGS)
LDFLAGS = $(CORE_PATH)/libdmlc.a $(DMLC_LDFLAGS) $(PS_LDFLAGS) $(EXTRA_LDFLAGS)

all: ps core bin tool # test

clean:
	rm -rf build/*

bin: build/async_sgd

build/%.o: %.cc
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -MM -MT build/$*.o $< >build/$*.d
	$(CXX) $(CFLAGS) -c $< -o $@

%.pb.cc %.pb.h : %.proto
	${DEPS_PATH}/bin/protoc --cpp_out=. --proto_path=. $<

test: build/localizer_test build/spmv_test build/minibatch_iter_test \
build/arg_parser build/ftrl\

ROOTDIR = $(CURDIR)
core:
	make -C $(CORE_PATH) -j4 config=$(ROOTDIR)/$(config)
clean_core:
	make -C $(CORE_PATH) clean
ps:
	make -C $(PS_PATH) -j4 ps config=$(ROOTDIR)/$(config)
clean_ps:
	make -C $(PS_PATH) clean



build/async_sgd: build/proto/config.pb.o build/proto/workload.pb.o build/proto/data_format.pb.o \
build/sgd/async_sgd.o
	$(CXX) $(CFLAGS) $(filter %.o %.a, $^) $(PS_PATH)/build/libps.a $(LDFLAGS) -o $@

tool: build/text2rec build/print_rec

# build/minibatch_iter_test:
build/%: build/tool/%.o build/proto/data_format.pb.o
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

build/%: build/test/%.o build/proto/data_format.pb.o
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

-include build/*/*.d
-include build/*/*/*.d

build/ftrl: build/proto/config.pb.o build/proto/sys.pb.o build/test/ftrl.o build/proto/data_format.pb.o
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@
