CC      ?= gcc
CFLAGS  ?= -O3 -march=native -mavx -mf16c -std=c11 -Wall -Wextra \
           -Wno-unused-parameter -fno-math-errno -fno-trapping-math \
           -fstrict-aliasing -pthread -MMD -MP -Wno-unused-result
CPPFLAGS += -Iinclude -I. -D_GNU_SOURCE
LDFLAGS  += -lm -pthread

OUT         ?= uenr
MENU_OUT    ?= uenr-menu
BUILD       ?= build
EXTRA_CFLAGS ?=
CFLAGS      += $(EXTRA_CFLAGS)
SRC     := $(wildcard src/*.c) $(wildcard src/*/*.c)
LIB_SRC := $(filter-out src/main.c src/menu.c,$(SRC))
OBJ     := $(patsubst src/%.c,$(BUILD)/%.o,$(LIB_SRC))
LIB     := $(BUILD)/libuenr.a
TESTS   := $(patsubst tests/%.c,$(BUILD)/tests/%,$(wildcard tests/test_*.c))
DEPS    := $(OBJ:.o=.d) $(BUILD)/main.d $(BUILD)/menu.d $(TESTS:=.d)

.PHONY: all clean bench test check tools lto pgo-gen pgo-use

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

$(BUILD)/tests/%: tests/%.c $(LIB)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(CPPFLAGS) -MMD -MP -MF $@.d $< $(LIB) -o $@ $(LDFLAGS)

CPUBENCH_DIR ?= bench

bench: $(BUILD)/bench/gemmbench $(BUILD)/bench/kvbench $(BUILD)/bench/kvcheck $(CPUBENCH_DIR)/cpubench $(CPUBENCH_DIR)/threadbench

$(BUILD)/bench/%: bench/%.c $(LIB)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(CPPFLAGS) $< $(LIB) -o $@ $(LDFLAGS)

$(CPUBENCH_DIR)/cpubench: $(CPUBENCH_DIR)/cpubench.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

$(CPUBENCH_DIR)/threadbench: $(CPUBENCH_DIR)/threadbench.c
	$(CC) $(CFLAGS) -pthread $< -o $@ $(LDFLAGS)

test: $(TESTS)
	@set -e; for t in $(TESTS); do echo "== $$t"; ./$$t; done

check: all test bench
	@echo "== check: build + tests + benches ok"

-include $(DEPS)

clean:
	rm -rf build build-* prof uenr uenr-menu uenr-*
