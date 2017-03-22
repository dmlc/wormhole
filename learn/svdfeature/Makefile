include ../../make/ps_app.mk

all: build/svdfeature.dmlc

clean:
	rm -rf build *.pb.*

build/svdfeature.dmlc: build/config.pb.o build/svdfeature.o $(DMLC_SLIB)
	$(CXX) $(CFLAGS) $(filter %.o %.a, $^) $(LDFLAGS) -o $@
