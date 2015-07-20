# Makefile for wormhole project
ifneq ("$(wildcard ./config.mk)","")
	config = $(CURDIR)/config.mk
else
	config = $(CURDIR)/make/config.mk
endif

# number of threads
# NT=4

# the directory where deps are installed
DEPS_PATH=$(CURDIR)/deps

ROOTDIR = $(CURDIR)
REPOS = $(addprefix repo/, dmlc-core xgboost ps-lite rabit)

.PHONY: clean all test pull

all: xgboost kmeans linear fm

### repos and deps

# dmlc-core
repo/dmlc-core:
	git clone https://github.com/dmlc/dmlc-core $@
	ln -s repo/dmlc-core/tracker .

repo/dmlc-core/libdmlc.a: | repo/dmlc-core glog
	+	$(MAKE) -C repo/dmlc-core libdmlc.a config=$(config) DEPS_PATH=$(DEPS_PATH) CXX=$(CXX)

core: | repo/dmlc-core glog					# always build
	+	$(MAKE) -C repo/dmlc-core libdmlc.a config=$(config) DEPS_PATH=$(DEPS_PATH) CXX=$(CXX)

# ps-lite
repo/ps-lite:
	git clone https://github.com/dmlc/ps-lite $@

repo/ps-lite/build/libps.a: | repo/ps-lite deps
	+	$(MAKE) -C repo/ps-lite ps config=$(config) DEPS_PATH=$(DEPS_PATH) CXX=$(CXX)


ps-lite: | repo/ps-lite deps 	# awlays build
	+	$(MAKE) -C repo/ps-lite ps config=$(config) DEPS_PATH=$(DEPS_PATH) CXX=$(CXX)

# rabit
repo/rabit:
	git clone https://github.com/dmlc/rabit $@

repo/rabit/lib/librabit.a:  | repo/rabit
	+	$(MAKE) -C repo/rabit

rabit: repo/rabit/lib/librabit.a

# deps
include make/deps.mk

deps: gflags glog protobuf zmq lz4 cityhash

### toolkits

# xgboost
repo/xgboost:
	git clone https://github.com/dmlc/xgboost $@

repo/xgboost/xgboost: repo/dmlc-core/libdmlc.a | repo/xgboost
	+	$(MAKE) -C repo/xgboost config=$(config)

bin/xgboost.dmlc: repo/xgboost/xgboost
	cp $+ $@

xgboost: bin/xgboost.dmlc

# kmeans
learn/kmeans/kmeans.dmlc: learn/kmeans/kmeans.cc | repo/rabit/lib/librabit.a repo/dmlc-core/libdmlc.a
	+	$(MAKE) -C learn/kmeans kmeans.dmlc

bin/kmeans.dmlc: learn/kmeans/kmeans.dmlc
	cp $+ $@

kmeans: bin/kmeans.dmlc

learn/base/base.a:
	$(MAKE) -C learn/base DEPS_PATH=$(DEPS_PATH) CXX=$(CXX)

# linear
learn/linear/build/linear.dmlc: ps-lite core repo/ps-lite/build/libps.a repo/dmlc-core/libdmlc.a learn/base/base.a
	$(MAKE) -C learn/linear config=$(config) DEPS_PATH=$(DEPS_PATH) CXX=$(CXX)

bin/linear.dmlc: learn/linear/build/linear.dmlc
	cp $+ $@

linear: bin/linear.dmlc

# FM
learn/difacto/build/fm.dmlc: ps-lite core repo/ps-lite/build/libps.a repo/dmlc-core/libdmlc.a learn/base/base.a
	$(MAKE) -C learn/difacto config=$(config) DEPS_PATH=$(DEPS_PATH) CXX=$(CXX)

bin/fm.dmlc: learn/difacto/build/fm.dmlc
	cp $+ $@

fm: bin/fm.dmlc



pull:
	for prefix in $(REPOS); do \
		if [ -d $$prefix ]; then \
			cd $$prefix; git pull; cd $(ROOTDIR); \
		fi \
	done

clean:
	for prefix in $(REPOS); do \
		if [ -d $$prefix ]; then \
			$(MAKE) -C $$prefix clean; \
		fi \
	done
	rm -rf bin/*
