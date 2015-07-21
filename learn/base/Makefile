ifndef config
ifneq ("$(wildcard ../../config.mk)","")
config = ../../config.mk
else
config = ../../make/config.mk
endif
endif

include $(config)

ifndef DEPS_PATH
DEPS_PATH = ../../deps
endif

all: text2crb

clean:
	rm -rf *.o text2crb

CORE_PATH=../../repo/dmlc-core
include $(CORE_PATH)/make/dmlc.mk

INCLUDE=-I.. -I. -I$(CORE_PATH)/include -I$(CORE_PATH)/src -I$(DEPS_PATH)/include

CFLAGS  = -O3 -ggdb -Wall -std=c++11 $(INCLUDE)

%.o: %.cc
	$(CXX) $(CFLAGS) -MM -MT $*.o $< >$*.d
	$(CXX) $(CFLAGS) -O3 -c $< -o $@


text2crb: text2crb.o
	$(CXX) $(CFLAGS) $^ $(CORE_PATH)/libdmlc.a $(DMLC_LDFLAGS) $(addprefix $(DEPS_PATH)/lib/, libglog.a libgflags.a libcityhash.a liblz4.a) -o $@


-include *.d

# base.a: proto/data_format.pb.o
# 	ar crv $@ $(filter %.o, $?)

# %.pb.cc %.pb.h : %.proto
# 	${DEPS_PATH}/bin/protoc --cpp_out=. --proto_path=. $<
