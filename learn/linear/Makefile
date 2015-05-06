ifneq ("$(wildcard ../config.mk)","")
	config = ../../config.mk
else
	config = ../../make/config.mk
endif

include $(config)
include ../../dmlc-core/make/dmlc.mk

DEPS_PATH=/usr/local/

CFLAGS  = -O3 -ggdb -Wall -std=c++11 -I../../dmlc-core/include -I../../dmlc-core/src -I./ -fopenmp
LDFLAGS = ../../dmlc-core/libdmlc.a -pthread -lm

all: build/ftrl # build/linear test

clean:
	rm -rf build

build/linear:


build/%.o: %.cc
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -MM -MT build/$*.o $< >build/$*.d
	$(CXX) $(CFLAGS) -c $< -o $@

%.pb.cc %.pb.h : %.proto
	${DEPS_PATH}/bin/protoc --cpp_out=. --proto_path=. $<

test: build/localizer_test build/spmv_test build/minibatch_iter_test \
build/arg_parser\

build/ftrl: build/sgd/ftrl.o build/proto/linear.pb.o -lprotobuf -lglog -lgflags
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

build/%: build/test/%.o
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

-include build/*/*.d
-include build/*/*/*.d
