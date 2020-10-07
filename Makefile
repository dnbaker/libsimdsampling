.PHONY=clean all


CXX?=g++


CXXFLAGS+=-march=native -O3 -I.

ifdef SLEEF_DIR
CXXFLAGS+= -L$(SLEEF_DIR)/lib
endif


all: libsimdsampling.a libsimdsampling.so libsimdsampling-st.so

simdsampling.cpp: simdsampling.h

simdsampling.o: simdsampling.cpp
	$(CXX) $(CXXFLAGS) -c -fPIC $< -o $@

libsimdsampling.a: simdsampling.o
	ar rcs $@ $<

libsimdsampling.so: simdsampling.o
	$(CXX) $(CXXFLAGS) -shared -o $@ $< -lsleef -fopenmp

libsimdsampling-st.so: simdsampling.o
	$(CXX) $(CXXFLAGS) -shared -o $@ $< -lsleef

test: test.cpp libsimdsampling.so
	$(CXX) $(CXXFLAGS) -L. -lsimdsampling $< -o $@

clean:
	rm -f libsimdsampling.a simdsampling.o libsimdsampling.so
