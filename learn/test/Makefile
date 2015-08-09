include ../../make/ps_app.mk

all: build/iter_solver_test

clean:
	rm -rf build

build/iter_solver_test: build/iter_solver_test.o $(DMLC_SLIB)
	$(CXX) $(CFLAGS) $(filter %.o %.a, $^) $(LDFLAGS) -o $@
