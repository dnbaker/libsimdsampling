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

all: libsimdsampling.a libsimdsampling.so libsimdsampling-st.so test test-st ctest ctest-st ftest ftest-st ktest ktest-st

run_tests: all
	./test && ./test-st && ./ctest && ./ctest-st && ./ftest && ./ftest-st && ./ktest && ./ktest-st

simdsampling.cpp: simdsampling.h

simdsampling-st.o: simdsampling.cpp libsleef-dyn
	$(CXX) $(CXXFLAGS) -c -fPIC $< -o $@ -lsleef

simdsampling.o: simdsampling.cpp libsleef-dyn
	$(CXX) $(CXXFLAGS) -c -fPIC $< -o $@ -fopenmp -lsleef

libsimdsampling-st.a: simdsampling-st.o
	$(AR) rcs $@ $<

libsimdsampling.a: simdsampling.o
	$(AR) rcs $@ $< $(SLEEFARG)

libsimdsampling.so: simdsampling.o
	$(CXX) $(CXXFLAGS) -shared -o $@ $< -lsleef -fopenmp -fPIC

libsimdsampling-st.so: simdsampling-st.o
	$(CXX) $(CXXFLAGS) -shared -o $@ $< -lsleef -fPIC -lsleef

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
	ls sleef 2>/dev/null || git clone https://github.com/shibatch/sleef

sleef/dynbuild: sleef
	ls sleef/dynbuild 2>/dev/null || mkdir sleef/dynbuild
sleef/build: sleef
	ls sleef/build 2>/dev/null || mkdir sleef/build

libsleef.a: sleef/build
	ls libsleef.a 2>/dev/null || (cd $< && $(CMAKE) .. -DBUILD_SHARED_LIBS=0 && $(MAKE) &&  cp lib/libsleef.a lib/libsleefdft.a ../..)

libsleef-dyn: sleef/dynbuild
	ls libsleef*so 2>/dev/null || ls libsleef*dylib 2>/dev/null || (cd sleef/dynbuild && echo "about to cmake " &&  $(CMAKE) .. -DBUILD_SHARED_LIBS=1 && $(MAKE) && (cp lib/libsleef*dylib ../.. 2>/dev/null || cp lib/libsleef*so ../.. 2>/dev/null))
clean:
	rm -f libsimdsampling.a simdsampling.o libsimdsampling.so libsimdsampling-st.so test test-st simdsampling-st.o
