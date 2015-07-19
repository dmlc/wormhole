include ../../make/ps_app.mk

all: build/linear.dmlc

clean:
	rm -rf build *.pb.*

build/linear.dmlc: build/config.pb.o build/linear.o $(DMLC_SLIB)
	$(CXX) $(CFLAGS) $(filter %.o %.a, $^) $(LDFLAGS) -o $@
