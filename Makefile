CC      ?= gcc
CFLAGS  ?= -O3 -march=native -mavx -mf16c -std=c11 -Wall -Wextra \
           -Wno-unused-parameter -fno-math-errno -fno-trapping-math \
           -fstrict-aliasing -pthread -MMD -MP -Wno-unused-result
CPPFLAGS += -Iinclude -I. -D_GNU_SOURCE
LDFLAGS  += -lm -pthread

BUILD   := build
SRC     := $(wildcard src/*.c) $(wildcard src/*/*.c)
LIB_SRC := $(filter-out src/main.c src/menu.c,$(SRC))
OBJ     := $(patsubst src/%.c,$(BUILD)/%.o,$(LIB_SRC))
LIB     := $(BUILD)/libuenr.a
TESTS   := $(patsubst tests/%.c,$(BUILD)/tests/%,$(wildcard tests/test_*.c))
DEPS    := $(OBJ:.o=.d) $(BUILD)/main.d $(BUILD)/menu.d $(TESTS:=.d)

.PHONY: all clean bench test check tools

all: uenr uenr-menu

uenr: $(BUILD)/main.o $(LIB)
	$(CC) $^ -o $@ $(LDFLAGS)

uenr-menu: $(BUILD)/menu.o $(LIB)
	$(CC) $^ -o $@ $(LDFLAGS)

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

bench: $(BUILD)/bench/gemmbench $(CPUBENCH_DIR)/cpubench $(CPUBENCH_DIR)/threadbench

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
	rm -rf $(BUILD) uenr uenr-menu
