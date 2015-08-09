include ../../make/ps_app.mk
include build.mk

all: $(TEST)

clean:
	rm -rf build

%: %.o $(DMLC_SLIB)
	$(CXX) $(CFLAGS) $(filter %.o %.a, $^) $(LDFLAGS) -o $@
