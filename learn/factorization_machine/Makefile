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

INC=-I./ -I../ -I$(PS_PATH)/src -I$(CORE_PATH)/include -I$(CORE_PATH)/src -I$(DEPS_PATH)/include
CFLAGS  = -O3 -ggdb -Wall -std=c++11 $(DMLC_CFLAGS) $(PS_CFLAGS) $(INC) $(EXTRA_CFLAGS)
LDFLAGS = $(CORE_PATH)/libdmlc.a $(DMLC_LDFLAGS) $(PS_LDFLAGS) $(EXTRA_LDFLAGS)

.PHONY: all

all: base ps core bin # tool # test

clean:
	rm -rf build/*

bin: build/fm.dmlc

build/%.o: %.cc
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -MM -MT build/$*.o $< >build/$*.d
	$(CXX) $(CFLAGS) -c $< -o $@

%.pb.cc %.pb.h : %.proto
	${DEPS_PATH}/bin/protoc --cpp_out=. --proto_path=. $<

ROOTDIR = $(CURDIR)
core:
	make -C $(CORE_PATH) -j4 config=$(ROOTDIR)/$(config)
clean_core:
	make -C $(CORE_PATH) clean
ps:
	make -C $(PS_PATH) -j4 ps config=$(ROOTDIR)/$(config)
clean_ps:
	make -C $(PS_PATH) clean

base:
	make -C ../base -j4 all config=$(ROOTDIR)/$(config)

build/fm.dmlc: build/proto/config.pb.o build/fm.o ../base/base.a $(PS_PATH)/build/libps.a
	$(CXX) $(CFLAGS) $(filter %.o %.a, $^) $(LDFLAGS) -o $@

# tool: build/text2rec build/print_rec

# build/minibatch_iter_test:
# build/%: build/tool/%.o build/proto/data_format.pb.o
# 	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

# build/%: build/test/%.o build/proto/data_format.pb.o
# 	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

-include build/*/*.d
-include build/*/*/*.d
