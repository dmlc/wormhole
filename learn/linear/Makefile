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
CFLAGS  = -O3 -ggdb -Wall -std=c++11 $(DMLC_CFLAGS) $(PS_CFLAGS) $(INC)
LDFLAGS = $(CORE_PATH)/libdmlc.a $(DMLC_LDFLAGS) $(PS_LDFLAGS) #-pthread -lm

all: core ps bin # test

clean:
	rm -rf build/*

clean_deps:
	make -C $(CORE_PATH) clean
	make -C $(PS_PATH) clean

bin: build/ftrl build/async_sgd

build/%.o: %.cc
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -MM -MT build/$*.o $< >build/$*.d
	$(CXX) $(CFLAGS) -c $< -o $@

%.pb.cc %.pb.h : %.proto
	${DEPS_PATH}/bin/protoc --cpp_out=. --proto_path=. $<

test: build/localizer_test build/spmv_test build/minibatch_iter_test \
build/arg_parser\

ROOTDIR = $(CURDIR)
core: $(CORE_PATH)/libdmlc.a
$(CORE_PATH)/libdmlc.a:
	make -C $(CORE_PATH) -j4 config=$(ROOTDIR)/$(config)

ps: $(PS_PATH)/build/libps.a
$(PS_PATH)/build/libps.a:
	make -C $(PS_PATH) -j4 ps config=$(ROOTDIR)/$(config)

build/async_sgd: build/proto/config.pb.o build/proto/sys.pb.o \
build/sgd/async_sgd.o \
$(PS_PATH)/build/libps.a
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

build/ftrl: build/proto/config.pb.o build/proto/sys.pb.o build/sgd/ftrl.o
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

build/%: build/test/%.o
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

-include build/*/*.d
-include build/*/*/*.d
