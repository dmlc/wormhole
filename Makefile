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

all: xgboost kmeans linear difacto svdfeature tool

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
	cd $@; git checkout tags/v1; cd ..

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
	+	$(MAKE) -C learn/kmeans kmeans.dmlc DEPS_PATH=$(DEPS_PATH) CXX=$(CXX)

bin/kmeans.dmlc: learn/kmeans/kmeans.dmlc
	cp $+ $@

kmeans: bin/kmeans.dmlc

# lbfgs
learn/lbfgs-linear/lbfgs.dmlc: learn/lbfgs-linear/lbfgs.cc | repo/rabit/lib/librabit.a repo/dmlc-core/libdmlc.a
	+	$(MAKE) -C learn/lbfgs-linear lbfgs.dmlc DEPS_PATH=$(DEPS_PATH) CXX=$(CXX)

bin/lbfgs.dmlc: learn/lbfgs-linear/lbfgs.dmlc
	cp $+ $@

lbfgs: bin/lbfgs.dmlc

# linear
learn/linear/build/linear.dmlc: ps-lite core repo/ps-lite/build/libps.a repo/dmlc-core/libdmlc.a
	$(MAKE) -C learn/linear config=$(config) DEPS_PATH=$(DEPS_PATH) CXX=$(CXX)

bin/linear.dmlc: learn/linear/build/linear.dmlc
	cp $+ $@

linear: bin/linear.dmlc

# FM
learn/difacto/build/difacto.dmlc: ps-lite core repo/ps-lite/build/libps.a repo/dmlc-core/libdmlc.a
	$(MAKE) -C learn/difacto config=$(config) DEPS_PATH=$(DEPS_PATH) CXX=$(CXX)

bin/difacto.dmlc: learn/difacto/build/difacto.dmlc
	cp $+ $@

difacto: bin/difacto.dmlc

# svdfeature
learn/svdfeature/build/svdfeature.dmlc: ps-lite core repo/ps-lite/build/libps.a repo/dmlc-core/libdmlc.a
	$(MAKE) -C learn/svdfeature config=$(config) DEPS_PATH=$(DEPS_PATH) CXX=$(CXX)

bin/svdfeature.dmlc: learn/svdfeature/build/svdfeature.dmlc
	cp $+ $@

svdfeature: bin/svdfeature.dmlc	

# tools

bin/convert.dmlc:
	$(MAKE) -C learn/tool convert config=$(config) DEPS_PATH=$(DEPS_PATH) CXX=$(CXX)
	cp learn/tool/convert $@

tool: bin/convert.dmlc

# test
include learn/test/build.mk

learn/test/%: ps-lite core repo/ps-lite/build/libps.a repo/dmlc-core/libdmlc.a
	$(MAKE) -C learn/test $* config=$(config) DEPS_PATH=$(DEPS_PATH) CXX=$(CXX)

test: $(addprefix learn/test/, $(TEST))


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
	rm -rf bin/*.dmlc
