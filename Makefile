# Makefile for wormhole project

ifneq ("$(wildcard ./config.mk)","")
	config = config.mk
else
	config = make/config.mk
endif

ROOTDIR = $(CURDIR)
REPOS = dmlc-core repo/xgboost
BIN = xgboost.dmlc

.PHONY: clean all test pull

all: xgboost.dmlc

dmlc-core:
	git clone https://github.com/dmlc/dmlc-core; cd $(ROOTDIR)

dmlc-core/libdmlc.a: | dmlc-core
	+	cd dmlc-core; make libdmlc.a config=$(ROOTDIR)/$(config); cd $(ROOTDIR)

repo/xgboost:
	cd repo; git clone https://github.com/dmlc/xgboost; cd $(ROOTDIR)

repo/xgboost/xgboost: dmlc-core/libdmlc.a | repo/xgboost dmlc-core
	+	cd repo/xgboost; make dmlc=$(ROOTDIR)/dmlc-core config=$(ROOTDIR)/$(config)

xgboost.dmlc: repo/xgboost/xgboost
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
