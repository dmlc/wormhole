# Makefile for wormhole project
ifneq ("$(wildcard ./config.mk)","")
	config = config.mk
else
	config = make/config.mk
endif

ROOTDIR = $(CURDIR)
REPOS = dmlc-core repo/xgboost
CPBIN = xgboost.dmlc kmeans.dmlc

.PHONY: clean all test pull

all: xgboost.dmlc kmeans.dmlc

# dmlc-core
dmlc-core:
	git clone https://github.com/dmlc/dmlc-core; cd $(ROOTDIR)

dmlc-core/libdmlc.a: | dmlc-core
	+	cd dmlc-core; make libdmlc.a config=$(ROOTDIR)/$(config); cd $(ROOTDIR)

# xgboost
repo/xgboost:
	cd repo; git clone https://github.com/dmlc/xgboost; cd $(ROOTDIR)

repo/xgboost/xgboost: dmlc-core/libdmlc.a | repo/xgboost dmlc-core
	+	cd repo/xgboost; make dmlc=$(ROOTDIR)/dmlc-core config=$(ROOTDIR)/$(config)

# parameter server
repo/ps-lite:
	git clone https://github.com/dmlc/ps-lite repo/ps-lite

# rabit
repo/rabit:
	cd repo; git clone https://github.com/dmlc/rabit; cd $(ROOTDIR)

repo/rabit/lib/librabit.a:  | repo/rabit
	+	cd repo/rabit; make; cd $(ROOTDIR)

learn/kmeans/kmeans.dmlc: learn/kmeans/kmeans.cc |repo/rabit/lib/librabit.a dmlc-core/libdmlc.a
	+	cd learn/kmeans;make kmeans.dmlc; cd $(ROOTDIR)

learn/linear/build/linear.dmlc:
	make -C learn/linear

learn/factorization_machine/buide/fm.dmlc:
	make -C learn/factorization_machine

deps:
	make/build_deps.sh

# toolkits
xgboost.dmlc: repo/xgboost/xgboost
	cp $+ $@

kmeans.dmlc: learn/kmeans/kmeans.dmlc
	cp $+ $@

linear.dmlc: learn/linear/build/linear.dmlc
	cp $+ $@

fm.dmlc: learn/factorization_machine/buide/fm.dmlc
	cp $+ $@

pull:
	for prefix in $(REPOS); do \
		if [ -d $$prefix ]; then \
			cd $$prefix; git pull; cd $(ROOTDIR); \
		fi \
	done

clean:
	for prefix in $(REPOS); do \
		if [ -d $$prefix ]; then \
			cd $$prefix; make clean; cd $(ROOTDIR); \
		fi \
	done
	rm -rf $(BIN) *~ */*~
