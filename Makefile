.PHONY=clean all


CXX?=g++


CXXFLAGS+=-march=native -O3 -I.

ifdef SLEEF_DIR
CXXFLAGS+= -L$(SLEEF_DIR)/lib
endif


all: libsimdsampling.a libsimdsampling.so libsimdsampling-st.so test test-st

simdsampling.cpp: simdsampling.h

simdsampling-st.o: simdsampling.cpp
	$(CXX) $(CXXFLAGS) -c -fPIC $< -o $@

simdsampling.o: simdsampling.cpp
	$(CXX) $(CXXFLAGS) -c -fPIC $< -o $@ -fopenmp

libsimdsampling-st.a: simdsampling-st.o
	ar rcs $@ $<

libsimdsampling.a: simdsampling.o
	ar rcs $@ $<

libsimdsampling.so: simdsampling.o
	$(CXX) $(CXXFLAGS) -shared -o $@ $< -lsleef -fopenmp

libsimdsampling-st.so: simdsampling-st.o
	$(CXX) $(CXXFLAGS) -shared -o $@ $< -lsleef

test: test.cpp libsimdsampling.so
	$(CXX) $(CXXFLAGS) -L. -lsimdsampling $< -o $@ -fopenmp

test-st: test.cpp libsimdsampling-st.so
	$(CXX) $(CXXFLAGS) -L. -lsimdsampling-st $< -o $@

clean:
	rm -f libsimdsampling.a simdsampling.o libsimdsampling.so libsimdsampling-st.so test test-st simdsampling-st.o
