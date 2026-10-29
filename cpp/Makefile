# Build: make          (g++ or clang++ with C++17; no external libraries, only pthreads)
CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O3 -march=native -Wall -Wextra -pthread
SRC      := src/main.cpp src/library.cpp src/template_search.cpp src/beam.cpp src/certify.cpp src/race.cpp src/columns.cpp
HDR      := src/library.hpp src/template_search.hpp src/beam.hpp src/certify.hpp src/race.hpp src/pool.hpp src/env.hpp src/gadget_tables.inc

qcpt: $(SRC) $(HDR)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC)

clean:
	rm -f qcpt

.PHONY: clean
