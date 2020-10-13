.PHONY=clean all


CXX?=g++
CC?=gcc

AR?=gcc-ar

WARNINGS=-Wall -Wextra -Wno-ignored-qualifiers
EXTRA?=
CFLAGS+=-march=native -O3 -I. $(WARNINGS) $(EXTRA)
CXXFLAGS+=-march=native -O3 -I. -std=c++11 $(WARNINGS) $(EXTRA)

ifdef SLEEF_DIR
CXXFLAGS+= -L$(SLEEF_DIR)/lib
endif

INCLUDE_PATHS+=simdpcg/include
LINK_PATHS+=

INCLUDE=$(patsubst %,-I%,$(INCLUDE_PATHS))
LINK=$(patsubst %,-L%,$(LINK_PATHS))

CXXFLAGS+=$(INCLUDE) $(LINK)
CFLAGS+=$(INCLUDE) $(LINK)

all: libsimdsampling.a libsimdsampling.so libsimdsampling-st.so test test-st ctest ctest-st ftest ftest-st

simdsampling.cpp: simdsampling.h

simdsampling-st.o: simdsampling.cpp
	$(CXX) $(CXXFLAGS) -c -fPIC $< -o $@

simdsampling.o: simdsampling.cpp
	$(CXX) $(CXXFLAGS) -c -fPIC $< -o $@ -fopenmp

libsimdsampling-st.a: simdsampling-st.o
	$(AR) rcs $@ $<

libsimdsampling.a: simdsampling.o
	$(AR) rcs $@ $<

libsimdsampling.so: simdsampling.o
	$(CXX) $(CXXFLAGS) -shared -o $@ $< -lsleef -fopenmp

libsimdsampling-st.so: simdsampling-st.o
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

clean:
	rm -f libsimdsampling.a simdsampling.o libsimdsampling.so libsimdsampling-st.so test test-st simdsampling-st.o
