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


CORE_PATH=../../repo/dmlc-core
include $(CORE_PATH)/make/dmlc.mk

INCLUDE += -I.. -I$(CORE_PATH)/include -I$(CORE_PATH)/src -I$(DEPS_PATH)/include

CFLAGS  += -O3 -ggdb -Wall -std=c++11 $(INCLUDE) $(DMLC_CFLAGS)

LDFLAGS += $(CORE_PATH)/libdmlc.a $(DMLC_LDFLAGS) $(addprefix $(DEPS_PATH)/lib/, libglog.a libgflags.a libcityhash.a liblz4.a)

all: text2crb convert

clean:
	rm -rf *.o text2crb

%.o: %.cc
	$(CXX) $(CFLAGS) -MM -MT $*.o $< >$*.d
	$(CXX) $(CFLAGS) -O3 -c $< -o $@


text2crb: text2crb.o
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

convert: convert.o
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

-include *.d
