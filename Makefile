# Target profile: the shipped machine is an Intel i5-3210M (Ivy Bridge).
# Ivy Bridge implies AVX1 + F16C and disables AVX2/FMA in GCC and Clang.
# Override MARCH=native only when building for the machine you run on.
MARCH   ?= ivybridge
CC      ?= gcc
CFLAGS  ?= -O3 -march=$(MARCH) -mavx -mf16c -std=c11 -Wall -Wextra \
           -Wno-unused-parameter -fno-math-errno -fno-trapping-math \
           -fstrict-aliasing -pthread -MMD -MP -Wno-unused-result
CPPFLAGS += -Iinclude -I. -D_GNU_SOURCE
LDFLAGS  += -lm -pthread

OUT         ?= uenr
MENU_OUT    ?= uenr-menu
BUILD       ?= build
EXTRA_CFLAGS ?=
CFLAGS      += $(EXTRA_CFLAGS)
# GitHub Actions overrides CFLAGS with its own -march=native line.  Keep the
# language, warning and dependency flags required by this build even then.
override CFLAGS += -std=c11 -Wall -Wextra -Wno-unused-parameter \
                   -fno-math-errno -fno-trapping-math -fstrict-aliasing \
                   -pthread -MMD -MP -Wno-unused-result
SRC_ALL := $(wildcard src/*.c) $(wildcard src/*/*.c)
# AVX2+FMA kernel copies are compiled separately with -mavx2 -mfma and are
# selected at runtime; everything else keeps the Ivy Bridge (AVX1) flags.
AVX2_SRC := $(wildcard src/*/*_avx2.c)
SRC     := $(filter-out $(AVX2_SRC),$(SRC_ALL))
LIB_SRC := $(filter-out src/main.c src/menu.c,$(SRC))
OBJ     := $(patsubst src/%.c,$(BUILD)/%.o,$(LIB_SRC)) \
           $(patsubst src/%.c,$(BUILD)/%.o,$(AVX2_SRC))
AVX2_OBJ := $(patsubst src/%.c,$(BUILD)/%.o,$(AVX2_SRC))
LIB     := $(BUILD)/libuenr.a
TESTS   := $(patsubst tests/%.c,$(BUILD)/tests/%,$(wildcard tests/test_*.c))
DEPS    := $(OBJ:.o=.d) $(BUILD)/main.d $(BUILD)/menu.d $(TESTS:=.d)

.PHONY: all clean bench test check tools lto pgo-gen pgo-use native check-isa smoke

all: $(OUT) $(MENU_OUT)

$(OUT): $(BUILD)/main.o $(LIB)
	$(CC) $^ -o $@ $(LDFLAGS) $(EXTRA_CFLAGS)

$(MENU_OUT): $(BUILD)/menu.o $(LIB)
	$(CC) $^ -o $@ $(LDFLAGS) $(EXTRA_CFLAGS)

lto:
	$(MAKE) BUILD=build-lto OUT=uenr-lto MENU_OUT=uenr-menu-lto EXTRA_CFLAGS="-flto" all

pgo-gen:
	$(MAKE) BUILD=build-pgo OUT=uenr-pgo-gen MENU_OUT=uenr-menu-pgo-gen EXTRA_CFLAGS="-fprofile-generate=$(CURDIR)/prof" all

pgo-use:
	$(MAKE) BUILD=build-pgo2 OUT=uenr-pgo MENU_OUT=uenr-menu-pgo EXTRA_CFLAGS="-fprofile-use=$(CURDIR)/prof -fprofile-correction" all

$(LIB): $(OBJ)
	@mkdir -p $(dir $@)
	$(AR) rcs $@ $^

$(BUILD)/%.o: src/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(CPPFLAGS) -c $< -o $@

# Runtime-dispatched AVX2+FMA kernels.  x86-64-v3 (not native) keeps the
# object portable across the heterogeneous GitHub runner hosts.
AVX2_CFLAGS = $(filter-out -march=%,$(CFLAGS)) -march=x86-64-v3 -mavx2 -mfma
$(AVX2_OBJ): $(BUILD)/%.o: src/%.c
	@mkdir -p $(dir $@)
	$(CC) $(AVX2_CFLAGS) $(CPPFLAGS) -c $< -o $@

$(BUILD)/tests/%: tests/%.c $(LIB)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(CPPFLAGS) -MMD -MP -MF $@.d $< $(LIB) -o $@ $(LDFLAGS)

CPUBENCH_DIR ?= bench

bench: $(BUILD)/bench/gemmbench $(BUILD)/bench/kvbench $(BUILD)/bench/kvcheck $(BUILD)/bench/kvppl $(CPUBENCH_DIR)/cpubench $(CPUBENCH_DIR)/threadbench

$(BUILD)/bench/%: bench/%.c $(LIB)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(CPPFLAGS) $< $(LIB) -o $@ $(LDFLAGS)

$(CPUBENCH_DIR)/cpubench: $(CPUBENCH_DIR)/cpubench.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

$(CPUBENCH_DIR)/threadbench: $(CPUBENCH_DIR)/threadbench.c
	$(CC) $(CFLAGS) -pthread $< -o $@ $(LDFLAGS)

test: $(TESTS)
	@set -e; for t in $(TESTS); do echo "== $$t"; ./$$t; done

# Build explicitly for the machine running the build (may emit AVX2/FMA).
native:
	$(MAKE) MARCH=native all

# Assert the target build contains no AVX2/FMA instructions.  Ivy Bridge
# supports AVX1 + F16C only; accidentally requiring AVX2/FMA would crash it.
# The AVX1 objects (everything except the explicitly dispatched *_avx2.o
# files) must remain Ivy Bridge-clean.  The AVX2 objects are allowed -- and
# expected -- to contain FMA and are only reached after a runtime CPU check.
check-isa: $(OUT) $(MENU_OUT)
	@if command -v objdump >/dev/null 2>&1; then \
	  AVX1_OBJ="$(filter-out $(AVX2_OBJ),$(OBJ)) $(BUILD)/main.o $(BUILD)/menu.o"; \
	  if objdump -d $$AVX1_OBJ | grep -Eq \
	     '\b(vfmadd|vfmsub|vfnmadd|vfnmsub|vpermd|vpermq|vperm2i128|vpbroadcastd|vpbroadcastb|vpbroadcastw|vinserti128|vextracti128|vpgather|vpsllv|vpsrav)'; then \
	    echo "ERROR: AVX1 object contains AVX2/FMA instructions"; exit 1; \
	  fi; \
	  if ! objdump -d $(AVX2_OBJ) | grep -q '\bvfmadd'; then \
	    echo "ERROR: AVX2 kernel objects contain no FMA instructions"; exit 1; \
	  fi; \
	  echo "== ISA check: AVX1 objects clean; AVX2 objects use FMA"; \
	else echo "== ISA check skipped (objdump unavailable)"; fi

check: all test bench check-isa
	@echo "== check: build + tests + benches + ISA ok"

# End-to-end pipeline in a temporary directory (real data/checkpoints untouched).
smoke: all
	scripts/smoke.sh ./$(OUT)

-include $(DEPS)

clean:
	rm -rf build build-* prof uenr uenr-menu uenr-*
