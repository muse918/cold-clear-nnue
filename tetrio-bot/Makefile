CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -march=native -Wall -Wextra -Wpedantic
CPPFLAGS += -I. -Ithird_party
LDFLAGS ?=
export TMPDIR := $(abspath build)

CORE = engine.cpp geometry.cpp battle.cpp
SEARCH = $(CORE) search.cpp
EVAL = $(SEARCH) evaluator.cpp
PROTOCOL = $(EVAL) protocol.cpp
HEADERS = engine.hpp search.hpp evaluator.hpp protocol.hpp geometry_tables.hpp
MODEL ?= ../experiments/spatial-nnue/results/final/seed918/attention.snn
TETRIO_JS ?=

.PHONY: all test verify-official gameplay clean
all: build/tetrio-bot

build:
	mkdir -p build

build/%.o: %.cpp $(HEADERS) | build
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

build/evaluator.o: ../experiments/spatial-nnue/kernels.cpp
build/protocol.o build/main.o build/protocol_tests.o: third_party/nlohmann/json.hpp

build/tetrio-bot: $(PROTOCOL:%.cpp=build/%.o) build/main.o
	$(CXX) $^ $(LDFLAGS) -o $@

build/geometry_tests: $(CORE:%.cpp=build/%.o) build/geometry_tests.o
	$(CXX) $^ $(LDFLAGS) -o $@
build/battle_tests: $(CORE:%.cpp=build/%.o) build/battle_tests.o
	$(CXX) $^ $(LDFLAGS) -o $@
build/search_tests: $(SEARCH:%.cpp=build/%.o) build/search_tests.o
	$(CXX) $^ $(LDFLAGS) -o $@
build/evaluator_tests: $(EVAL:%.cpp=build/%.o) build/evaluator_tests.o
	$(CXX) $^ $(LDFLAGS) -o $@
build/protocol_tests: $(PROTOCOL:%.cpp=build/%.o) build/protocol_tests.o
	$(CXX) $^ $(LDFLAGS) -o $@
build/gameplay_bench: $(EVAL:%.cpp=build/%.o) build/gameplay_bench.o
	$(CXX) $^ $(LDFLAGS) -o $@
build/battle_probe: $(CORE:%.cpp=build/%.o) tests/rules/battle_probe.cpp third_party/nlohmann/json.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tests/rules/battle_probe.cpp $(CORE:%.cpp=build/%.o) $(LDFLAGS) -o $@

test: build/tetrio-bot build/geometry_tests build/battle_tests build/search_tests build/evaluator_tests build/protocol_tests
	build/geometry_tests
	build/battle_tests
	build/search_tests
	build/evaluator_tests $(MODEL)
	build/protocol_tests
	python3 tests/protocol_smoke.py build/tetrio-bot

verify-official: build/geometry_tests build/battle_probe
	test -n "$(TETRIO_JS)"
	node tests/geometry/verify_official_geometry.mjs "$(TETRIO_JS)" build/geometry_tests tests/geometry/official_geometry_report.json
	node tests/rules/verify_battle.mjs --official "$(TETRIO_JS)" --probe build/battle_probe --report tests/rules/battle_parity.json

gameplay: build/gameplay_bench
	build/gameplay_bench $(MODEL) > tests/gameplay_report.json

clean:
	rm -rf build

-include $(wildcard build/*.d)
