include ../../make/ps_app.mk

CFLAGS  = -O3 -ggdb -Wall -std=c++11 -I./ -I../ $(INCLUDE) $(DMLC_CFLAGS) $(PS_CFLAGS) $(EXTRA_CFLAGS)
LDFLAGS = $(CORE_PATH)/libdmlc.a $(DMLC_LDFLAGS) $(PS_LDFLAGS) $(EXTRA_LDFLAGS)

all: base ps core bin # tool # test

clean:
	rm -rf build/*

bin: build/fm.dmlc

build/%.o: %.cc
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) -MM -MT build/$*.o $< >build/$*.d
	$(CXX) $(CFLAGS) -c $< -o $@

%.pb.cc %.pb.h : %.proto
	${DEPS_PATH}/bin/protoc --cpp_out=. --proto_path=. $<


build/fm.dmlc: build/config.pb.o build/fm.o ../base/base.a $(PS_PATH)/build/libps.a
	$(CXX) $(CFLAGS) $(filter %.o %.a, $^) $(LDFLAGS) -o $@


-include build/*.d
