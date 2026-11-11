# Run from this directory. Benchmarks must be serial and CPU training must stop.
.NOTPARALLEL:
.SHELLFLAGS := -eu -c
CXX ?= g++
PYTHON ?= python3
NODE ?= node
CXXFLAGS ?= -O3 -DNDEBUG -std=c++17 -march=native
DATA_CXXFLAGS ?= -O3 -std=c++17
QUANT_CXXFLAGS ?= -O3 -std=c++17 -march=native
MODELS = cell cnn attention
SEEDS = 918 919 920
FINAL = results/final/seed918
ENCODER = $(if $(filter cell,$(1)),,--encoder-source $(FINAL)/$(1).encoder.f32)

.PHONY: all data train evaluate verify benchmark report clean
all: build/generate build/kernels build/quantized build/search_bench data/libmoves.so data/libkernels.so

build:
	mkdir -p build

build/generate: generate.cpp | build
	$(CXX) $(DATA_CXXFLAGS) $< -o $@

build/kernels: kernels.cpp | build
	$(CXX) $(CXXFLAGS) $< -o $@

build/quantized: quantized.cpp kernels.cpp | build
	$(CXX) $(QUANT_CXXFLAGS) $< -o $@

build/search_bench: search_bench.cpp generate.cpp kernels.cpp | build
	$(CXX) $(CXXFLAGS) $< -o $@

data/libmoves.so: generate.cpp
	mkdir -p data
	$(CXX) $(DATA_CXXFLAGS) -shared -fPIC -DSPATIAL_DATA_LIBRARY $< -o $@.pending
	mv $@.pending $@

data/libkernels.so: kernels.cpp
	mkdir -p data
	$(CXX) $(CXXFLAGS) -shared -fPIC -DSPATIAL_KERNEL_LIBRARY $< -o $@.pending
	mv $@.pending $@

data/train/records.bin: build/generate
	$< --out data/train --games 96 --steps 44 --seed 91820261111 --start-game 0

data/heldout/records.bin: build/generate
	$< --out data/heldout --games 16 --steps 80 --seed 91820261111 --start-game 96

data/stress/records.bin: stress.py train.py
	$(PYTHON) stress.py

data: data/train/records.bin data/heldout/records.bin data/stress/records.bin

train: all data
	$(PYTHON) train.py --data data/train/records.bin --out results/final/seed918 --models cell,cnn,attention,dense_cnn,transformer --epochs 14 --batch 512 --threads 4 --seed 918
	$(PYTHON) train.py --data data/train/records.bin --out results/final/seed919 --models cell,cnn,attention --epochs 14 --batch 512 --threads 4 --seed 919
	$(PYTHON) train.py --data data/train/records.bin --out results/final/seed920 --models cell,cnn,attention --epochs 14 --batch 512 --threads 4 --seed 920

evaluate: all data
	$(PYTHON) evaluate.py

# Supply the audited external client explicitly; it is not redistributed.
verify: all data
	test -n "$(TETRIO_JS)"
	$(NODE) rules/verify_row_events.mjs "$(TETRIO_JS)" > rules/row_event_results.json
	$(NODE) rules/validate_dataset.mjs data/train/records.bin "$(TETRIO_JS)" rules/dataset_parity.json --all
	$(NODE) rules/validate_dataset.mjs data/heldout/records.bin "$(TETRIO_JS)" rules/heldout_dataset_parity.json --all
	$(NODE) rules/validate_dataset.mjs data/stress/records.bin "$(TETRIO_JS)" rules/stress_parity.json --all
	$(foreach model,$(MODELS),build/kernels --data data/train/records.bin --model $(FINAL)/$(model).snn --output $(FINAL)/$(model)_native_validation.json --verify-only $(call ENCODER,$(model));)
	$(foreach model,$(MODELS),build/kernels --data data/heldout/records.bin --model $(FINAL)/$(model).snn --output $(FINAL)/$(model)_native_heldout_validation.json --verify-only $(call ENCODER,$(model));)
	$(foreach model,$(MODELS),build/kernels --data data/stress/records.bin --model $(FINAL)/$(model).snn --output $(FINAL)/$(model)_native_stress_validation.json --verify-only $(call ENCODER,$(model));)
	$(foreach model,$(MODELS),build/search_bench --data data/heldout/records.bin --model $(FINAL)/$(model).snn --output $(FINAL)/$(model)_search_validation.json --verify-only;)
	$(foreach seed,$(SEEDS),$(foreach model,$(MODELS),$(foreach bits,8 16,build/quantized --data data/heldout/records.bin --model results/final/seed$(seed)/$(model).snn --output results/final/seed$(seed)/$(model)_quantized_q$(bits).json --bits $(bits) --repeats 0;)))
	$(foreach model,$(MODELS),$(foreach bits,8 16,build/quantized --data data/stress/records.bin --model $(FINAL)/$(model).snn --output $(FINAL)/$(model)_quantized_q$(bits)_stress.json --bits $(bits) --repeats 0;))

# Regenerates timing reports. Do not run alongside training or other benchmarks.
benchmark: all data
	$(foreach model,$(MODELS),build/kernels --data data/train/records.bin --model $(FINAL)/$(model).snn --output $(FINAL)/$(model)_native_train_benchmark.json --repeats 9 $(call ENCODER,$(model));)
	$(foreach model,$(MODELS),build/kernels --data data/heldout/records.bin --model $(FINAL)/$(model).snn --output $(FINAL)/$(model)_native_heldout_benchmark.json --repeats 9 $(call ENCODER,$(model));)
	$(foreach model,$(MODELS),build/search_bench --data data/heldout/records.bin --model $(FINAL)/$(model).snn --output $(FINAL)/$(model)_search_timing.json --repeats 9;)
	$(foreach model,$(MODELS),$(foreach bits,8 16,build/quantized --data data/heldout/records.bin --model $(FINAL)/$(model).snn --output $(FINAL)/$(model)_quantized_q$(bits)_timing.json --bits $(bits) --repeats 9;))

report:
	$(PYTHON) summarize.py

clean:
	rm -rf build
