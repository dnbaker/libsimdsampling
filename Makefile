.PHONY=clean all


CXX?=g++
CC?=gcc

AR?=gcc-ar
CMAKE?=cmake

WARNINGS=-Wall -Wextra -Wno-ignored-qualifiers
EXTRA?=
CFLAGS+=-march=native -O3 -I. $(WARNINGS) $(EXTRA)
CXXFLAGS+=-march=native -O3 -I. -std=c++11 $(WARNINGS) $(EXTRA)

ifdef SLEEF_DIR
CXXFLAGS+= -L$(SLEEF_DIR)/lib
endif

INCLUDE_PATHS+=sleef/build/include
LINK_PATHS+=sleef/build/lib

INCLUDE=$(patsubst %,-I%,$(INCLUDE_PATHS))
LINK=$(patsubst %,-L%,$(LINK_PATHS))

CXXFLAGS+=$(INCLUDE) $(LINK)
CFLAGS+=$(INCLUDE) $(LINK)

SLEEFARG=libsleef.a

all: libsimdsampling.a libsimdsampling.so libsimdsampling-st.so test test-st ctest ctest-st ftest ftest-st

run_tests: all
	./test && ./test-st && ./ctest && ./ctest-st && ./ftest && ./ftest-st

simdsampling.cpp: simdsampling.h

simdsampling-st.o: simdsampling.cpp $(SLEEFARG)
	$(CXX) $(CXXFLAGS) -c -fPIC $< -o $@

simdsampling.o: simdsampling.cpp $(SLEEFARG)
	$(CXX) $(CXXFLAGS) -c -fPIC $< -o $@ -fopenmp

libsimdsampling-st.a: simdsampling-st.o $(SLEEFARG)
	$(AR) rcs $@ $< $(SLEEFARG)

libsimdsampling.a: simdsampling.o $(SLEEFARG)
	$(AR) rcs $@ $< $(SLEEFARG)

libsimdsampling.so: simdsampling.o $(SLEEFARG)
	$(CXX) $(CXXFLAGS) -shared -o $@ $< -lsleef -fopenmp

libsimdsampling-st.so: simdsampling-st.o $(SLEEFARG)
	$(CXX) $(CXXFLAGS) -shared -o $@ $< -lsleef

ftest: test.cpp libsimdsampling.so
	$(CXX) $(CXXFLAGS) -L. -lsimdsampling $< -o $@ -fopenmp -DFLOAT_TYPE=float

ftest-st: test.cpp libsimdsampling-st.so
	$(CXX) $(CXXFLAGS) -L. -lsimdsampling-st $< -o $@ -DFLOAT_TYPE=float

test: test.cpp libsimdsampling.so
	$(CXX) $(CXXFLAGS) -L. -lsimdsampling $< -o $@ -fopenmp

test-st: test.cpp libsimdsampling-st.so
	$(CXX) $(CXXFLAGS) -L. -lsimdsampling-st $< -o $@

ctest: ctest.c libsimdsampling.so
	$(CC) $(CFLAGS) -L. -lsimdsampling $< -o $@ -fopenmp

ctest-st: ctest.c libsimdsampling-st.so
	$(CC) $(CFLAGS) -L. -lsimdsampling-st $< -o $@

ktest: ktest.cpp libsimdsampling.so
	$(CXX) $(CXXFLAGS) -L. -lsimdsampling $< -o $@ -fopenmp

ktest-st: ktest.cpp libsimdsampling-st.so
	$(CXX) $(CXXFLAGS) -L. -lsimdsampling-st $< -o $@

sleef:
	ls sleef || git clone https://github.com/shibatch/sleef

libsleef.a: sleef
	(ls libsleef.a || (cd sleef && (mkdir build && cd build || cd build && make clean) && cd build && $(CMAKE) .. -DBUILD_SHARED_LIBS=0 && $(MAKE) && cp lib/libsleef.a lib/libsleefdft.a ../.. && cd ..)) && \
    ((ls sleef/build/lib/libsleef*so && ls sleef/build/lib/libsleef*dylib) || (cd sleef && mkdir -p build && cd build && make clean && $(CMAKE) .. -DBUILD_SHARED_LIBS=1 && $(MAKE) && cp lib/libsleef*dylib lib/libsleef*so ../.. && cd ..))
clean:
	rm -f libsimdsampling.a simdsampling.o libsimdsampling.so libsimdsampling-st.so test test-st simdsampling-st.o
