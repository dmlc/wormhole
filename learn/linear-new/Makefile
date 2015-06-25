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
	rm -rf
	rm -rf build/*

bin: build/linear.dmlc

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

build/linear.dmlc: build/config.pb.o build/linear.o ../base/base.a $(PS_PATH)/build/libps.a
	$(CXX) $(CFLAGS) $(filter %.o %.a, $^) $(LDFLAGS) -o $@


-include build/*.d
