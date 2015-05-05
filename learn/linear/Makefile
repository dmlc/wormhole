ifneq ("$(wildcard ../config.mk)","")
	config = ../../config.mk
else
	config = ../../make/config.mk
endif

include $(config)
include ../../dmlc-core/make/dmlc.mk

# THIRD_PATH=/Users/muli/work/third_party_gcc/

CFLAGS  = -O0 -ggdb -Wall -std=c++11 -I../../dmlc-core/include -I../../dmlc-core/src -I./ -fopenmp
LDFLAGS = ../../dmlc-core/libdmlc.a -pthread -lm

all: build/linear test

clean:
	rm -rf build

build/linear:


build/%.o: %.cc
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -MM -MT build/$*.o $< >build/$*.d
	$(CXX) $(CFLAGS) -c $< -o $@

# %.pb.cc %.pb.h : %.proto
# 	${THIRD_PATH}/bin/protoc --cpp_out=. --proto_path=. $<

test: build/localizer_test build/spmv_test build/minibatch_iter_test

build/%: build/test/%.o
	$(CXX) $(CFLAGS) $^ $(LDFLAGS) -o $@

-include build/*/*.d
-include build/*/*/*.d
