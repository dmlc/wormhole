# common make for parameter server applications

ifneq ("$(wildcard ../../config.mk)","")
	config = ../../config.mk
else
	config = ../../make/config.ps.mk
endif
include $(config)

CORE_PATH=../../dmlc-core
include $(CORE_PATH)/make/dmlc.mk

PS_PATH=../../repo/ps-lite
include $(PS_PATH)/make/ps.mk
ifndef DEPS_PATH
DEPS_PATH = $(PS_PATH)/deps
endif

.DEFAULT_GOAL := all

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

INCLUDE=-I$(PS_PATH)/src -I$(CORE_PATH)/include -I$(CORE_PATH)/src -I$(DEPS_PATH)/include
