include $(config)

all: base.a
clean:
	rm -rf base.a proto/*.pb.*

base.a: proto/data_format.pb.o
	ar crv $@ $(filter %.o, $?)

%.o: %.cc
	$(CXX) -I. -I$(DEPS_PATH)/include -O3 -c $< -o $@

%.pb.cc %.pb.h : %.proto
	${DEPS_PATH}/bin/protoc --cpp_out=. --proto_path=. $<
