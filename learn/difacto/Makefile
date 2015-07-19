include ../../make/ps_app.mk

all: build/fm.dmlc

clean:
	rm -rf build *.pb.*

build/fm.dmlc: build/config.pb.o build/fm.o $(DMLC_SLIB)
	$(CXX) $(CFLAGS) $(filter %.o %.a, $^) $(LDFLAGS) -o $@
