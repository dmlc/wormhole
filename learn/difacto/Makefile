include ../../make/ps_app.mk

all: build/difacto.dmlc

clean:
	rm -rf build *.pb.*

build/difacto.dmlc: build/config.pb.o build/difacto.o $(DMLC_SLIB)
	$(CXX) $(CFLAGS) $(filter %.o %.a, $^) $(LDFLAGS) -o $@
