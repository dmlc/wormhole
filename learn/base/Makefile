include $(config)

ifndef DEPS_PATH
DEPS_PATH = ../../repo/ps-lite/deps
endif

ifndef CXX
CXX = g++
endif

all: base.a
clean:
	rm -rf base.a proto/*.pb.*

base.a: proto/data_format.pb.o proto/workload.pb.o
	ar crv $@ $(filter %.o, $?)

%.o: %.cc
	$(CXX) -I. -O3 -c $< -o $@

%.pb.cc %.pb.h : %.proto
	${DEPS_PATH}/bin/protoc --cpp_out=. --proto_path=. $<
