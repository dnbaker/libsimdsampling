.PHONY=clean all


CXX?=g++
CC?=gcc

AR?=gcc-ar
CMAKE?=cmake

WARNINGS=-Wall -Wextra -Wno-ignored-qualifiers -Wno-unused-function
EXTRA?=
CFLAGS+=-march=native -O3 -I. $(WARNINGS) $(EXTRA) -pthread
CXXFLAGS+=-march=native -O3 -I. -std=c++11 $(WARNINGS) $(EXTRA) -pthread

ifdef SLEEF_DIR
CXXFLAGS+= -L$(SLEEF_DIR)/lib
endif

INCLUDE_PATHS+=sleef/build/include sleef/dynbuild/include
LINK_PATHS+=sleef/dynbuild/lib

INCLUDE=$(patsubst %,-I%,$(INCLUDE_PATHS))
LINK=$(patsubst %,-L%,$(LINK_PATHS))

CXXFLAGS+=$(INCLUDE) $(LINK)
CFLAGS+=$(INCLUDE) $(LINK)

SLEEFARG=libsleef.a

all: libsimdsampling.a libsimdsampling.so libsimdsampling-st.so libsimdsampling-st.a \
        test test-st ctest ctest-st ftest ftest-st ktest ktest-st \
     libargminmax.so libargminmax.a \
     argmintest cargredtest

DYNLIBS: libsimdsampling.so libsimdsampling-st.so libargminmax.so
STATICLIBS: libsimdsampling.a libsimdsampling-st.a libargminmax.a


libs: DYNLIBS STATICLIBS

run_tests: all
	./test && ./test-st && ./ctest && ./ctest-st && ./ftest && ./ftest-st && ./ktest && ./ktest-st \
           && ./argmintest  && ./cargredtest

simdsampling.cpp: simdsampling.h

simdsampling-approx-st.o: simdsampling.cpp libsleef-dyn
	$(CXX) $(CXXFLAGS) -c -fPIC $< -o $@ -DUSE_APPROX_LOG

simdsampling-st.o: simdsampling.cpp libsleef-dyn
	$(CXX) $(CXXFLAGS) -c -fPIC $< -o $@ -lsleef

simdsampling-approx.o: simdsampling.cpp libsleef-dyn
	$(CXX) $(CXXFLAGS) -c -fPIC $< -o $@ -fopenmp -DUSE_APPROX_LOG

simdsampling.o: simdsampling.cpp libsleef-dyn
	$(CXX) $(CXXFLAGS) -c -fPIC $< -o $@ -fopenmp -lsleef

libsimdsampling-st.a: simdsampling-st.o argminmax.o $(SLEEFARG)
	$(AR) rcs $@ $< argminmax.o $(SLEEFARG)

libargminmax.a: argminmax.o
	$(AR) rcs $@ $<

libsimdsampling.a: simdsampling.o argminmax.o libsleef.a
	$(AR) rcs $@ $< argminmax.o  $(SLEEFARG)

argminmax.o: argminmax.cpp
	$(CXX) $(CXXFLAGS) -c -fPIC $< -o $@ -fopenmp -lsleef

libargminmax.so: argminmax.o
	$(CXX) $(CXXFLAGS) -shared -o $@ $< -fopenmp -fPIC

libsimdsampling-approx.so: simdsampling-approx.o argminmax.o
	$(CXX) $(CXXFLAGS) -shared -o $@ $< argminmax.o -lsleef -fopenmp -fPIC

libsimdsampling-approx-st.so: simdsampling-approx-st.o argminmax.o
	$(CXX) $(CXXFLAGS) -shared -o $@ $< argminmax.o -fPIC -lsleef -fopenmp

libsimdsampling.so: simdsampling.o argminmax.o
	$(CXX) $(CXXFLAGS) -shared -o $@ $< argminmax.o -lsleef -fopenmp -fPIC

libsimdsampling-st.so: simdsampling-st.o argminmax.o
	$(CXX) $(CXXFLAGS) -shared -o $@ $< argminmax.o -lsleef -fPIC -lsleef -fopenmp

ftest: test.cpp libsimdsampling.so
	$(CXX) $(CXXFLAGS) -L. -lsimdsampling $< -o $@ -fopenmp -DFLOAT_TYPE=float

ftest-approx-st: test.cpp libsimdsampling-approx-st.so
	$(CXX) $(CXXFLAGS) -L. -lsimdsampling-approx-st $< -o $@ -DFLOAT_TYPE=float -lsleef

ftest-st: test.cpp libsimdsampling-st.so
	$(CXX) $(CXXFLAGS) -L. -lsimdsampling-st $< -o $@ -DFLOAT_TYPE=float

test-approx: test.cpp libsimdsampling-approx.so
	$(CXX) $(CXXFLAGS) -L. -lsimdsampling $< -o $@ -fopenmp

test: test.cpp libsimdsampling.so
	$(CXX) $(CXXFLAGS) -L. -lsimdsampling $< -o $@ -fopenmp

test-st: test.cpp libsimdsampling-st.so
	$(CXX) $(CXXFLAGS) -L. -lsimdsampling-st $< -o $@

test-approx-st: test.cpp libsimdsampling-approx-st.so
	$(CXX) $(CXXFLAGS) -L. -lsimdsampling-approx-st $< -o $@ -lsleef

ctest: ctest.c libsimdsampling.so
	$(CC) $(CFLAGS) -L. -lsimdsampling $< -o $@ -fopenmp

ctest-st: ctest.c libsimdsampling-st.so
	$(CC) $(CFLAGS) -L. -lsimdsampling-st $< -o $@

ktest: ktest.cpp libsimdsampling.so
	$(CXX) $(CXXFLAGS) -L. -lsimdsampling $< -o $@ -fopenmp

ktest-st: ktest.cpp libsimdsampling-st.so
	$(CXX) $(CXXFLAGS) -L. -lsimdsampling-st $< -o $@

argmintest: argmintest.cpp libargminmax.so
	$(CXX) $(CXXFLAGS) -L. -largminmax $< -o $@ -fopenmp
cargredtest: argmintest.cpp libargminmax.so
	$(CXX) $(CXXFLAGS) -L. -largminmax $< -o $@



sleef:
	ls sleef 2>/dev/null || git clone https://github.com/shibatch/sleef

sleef/dynbuild: sleef
	ls sleef/dynbuild 2>/dev/null || mkdir sleef/dynbuild
sleef/build: sleef
	ls sleef/build 2>/dev/null || mkdir sleef/build

libsleef.a: sleef/build
	ls libsleef.a 2>/dev/null || ((ls ../libsleef.a && cp ../libsleef.a .) || cd $< && $(CMAKE) .. -DBUILD_SHARED_LIBS=0 && $(MAKE) &&  cp lib/libsleef.a lib/libsleefdft.a ../..)

libsleef-dyn: sleef/dynbuild
	ls libsleef*so 2>/dev/null || ls libsleef*dylib 2>/dev/null || (cd sleef/dynbuild && echo "about to cmake " &&  $(CMAKE) .. -DBUILD_SHARED_LIBS=1 && $(MAKE) && (cp lib/libsleef*dylib ../.. 2>/dev/null || cp lib/libsleef*so ../.. 2>/dev/null))
clean:
	rm -f libsimdsampling.a simdsampling.o libsimdsampling.so libsimdsampling-st.so libsimdsampling-st.a test test-st simdsampling-st.o \
        libargminmax.so argminmax.o argmintest argmintest-st cargredtest cargredtest-st \
        ftest ftest-st ktest ktest-st ctest ctest-st libargminmax.a
