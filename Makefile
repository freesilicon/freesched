HOST_CC  ?= clang
HOST_CXX ?= clang++
HOST_AR  ?= ar

THIS_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
BUILD_DIR := $(THIS_DIR)/build

INCS := -I$(THIS_DIR)/include -I$(THIS_DIR)/src
CFLAGS   := -std=c11 -O2 -Wall -Wextra -Werror -I$(THIS_DIR)/include
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Werror -fno-exceptions -fno-rtti $(INCS)
LDFLAGS  := -lm -lpthread
ifeq ($(shell uname -s),Linux)
LDFLAGS  += -ldl
endif

OPENBUENOS_DIR ?=
ifneq ($(OPENBUENOS_DIR),)
OB_LIB := $(firstword $(wildcard $(OPENBUENOS_DIR)/build/host/libopenbuenos.a $(OPENBUENOS_DIR)/build/linux-arm64/libopenbuenos.a $(OPENBUENOS_DIR)/build/libopenbuenos.a))
ifeq ($(OB_LIB),)
$(error OPENBUENOS_DIR=$(OPENBUENOS_DIR) specified but libopenbuenos.a not found)
endif
CXXFLAGS += -DFREESCHED_HAVE_LIBOPENBUENOS=1 -I$(OPENBUENOS_DIR)/include
LDFLAGS  := $(OB_LIB) $(LDFLAGS)
endif

SRCS := \
	$(THIS_DIR)/src/fs_topology.cpp \
	$(THIS_DIR)/src/fs_dmabuf.cpp \
	$(THIS_DIR)/src/fs_router.cpp \
	$(THIS_DIR)/src/fs_executor.cpp \
	$(THIS_DIR)/src/fs_backend_openbuenos.cpp \
	$(THIS_DIR)/src/fs_backend_vulkan_dmabuf.cpp \
	$(THIS_DIR)/src/ggml_freesilicon.cpp \
	$(THIS_DIR)/src/pc_gguf.cpp \
	$(THIS_DIR)/src/pc_kv_ring.cpp \
	$(THIS_DIR)/src/pc_soa_neon.cpp \
	$(THIS_DIR)/src/pc_dmabuf_graph.cpp

HDRS := \
	$(THIS_DIR)/include/freesched.h \
	$(THIS_DIR)/include/ggml_freesilicon.h \
	$(THIS_DIR)/include/pocketcoder_runner.h \
	$(THIS_DIR)/src/fs_internal.h

OBJS := $(patsubst $(THIS_DIR)/src/%.cpp,$(BUILD_DIR)/obj/%.o,$(SRCS))

LIB        := $(BUILD_DIR)/libfreesched.a
TEST_C11   := $(BUILD_DIR)/test_c11_abi
TEST_CXX   := $(BUILD_DIR)/test_freesched
TEST_PC    := $(BUILD_DIR)/test_pocketcoder
BENCH_BIN  := $(BUILD_DIR)/bench_heterogeneous
SMOKE_BIN  := $(BUILD_DIR)/freesched_dmabuf_smoke
PC_BIN     := $(BUILD_DIR)/pocketcoder-run

.PHONY: all lib test twin-test smoke bench verify-headers audit verify clean

all: lib $(TEST_C11) $(TEST_CXX) $(TEST_PC) $(BENCH_BIN) $(SMOKE_BIN) $(PC_BIN)

lib: $(LIB)

$(BUILD_DIR)/obj/%.o: $(THIS_DIR)/src/%.cpp $(HDRS)
	@mkdir -p $(dir $@)
	$(HOST_CXX) $(CXXFLAGS) -c $< -o $@

$(LIB): $(OBJS)
	@mkdir -p $(dir $@)
	$(HOST_AR) rcs $@ $(OBJS)

$(TEST_C11): $(THIS_DIR)/tests/test_c11_abi.c $(LIB) $(THIS_DIR)/include/freesched.h $(THIS_DIR)/include/ggml_freesilicon.h $(THIS_DIR)/include/pocketcoder_runner.h
	@mkdir -p $(dir $@)
	$(HOST_CC) $(CFLAGS) -c $< -o $(BUILD_DIR)/obj/test_c11_abi.o
	$(HOST_CXX) $(BUILD_DIR)/obj/test_c11_abi.o $(LIB) -o $@ $(LDFLAGS)

$(TEST_CXX): $(THIS_DIR)/tests/test_freesched.cpp $(LIB) $(HDRS)
	@mkdir -p $(dir $@)
	$(HOST_CXX) $(CXXFLAGS) $< $(LIB) -o $@ $(LDFLAGS)

$(TEST_PC): $(THIS_DIR)/tests/test_pocketcoder.cpp $(LIB) $(HDRS)
	@mkdir -p $(dir $@)
	$(HOST_CXX) $(CXXFLAGS) $< $(LIB) -o $@ $(LDFLAGS)

$(BENCH_BIN): $(THIS_DIR)/tests/bench_heterogeneous.cpp $(LIB) $(HDRS)
	@mkdir -p $(dir $@)
	$(HOST_CXX) $(CXXFLAGS) $< $(LIB) -o $@ $(LDFLAGS)

$(SMOKE_BIN): $(THIS_DIR)/tools/freesched_dmabuf_smoke.cpp $(LIB) $(HDRS)
	@mkdir -p $(dir $@)
	$(HOST_CXX) $(CXXFLAGS) $< $(LIB) -o $@ $(LDFLAGS)

$(PC_BIN): $(THIS_DIR)/tools/pocketcoder_run.cpp $(LIB) $(HDRS)
	@mkdir -p $(dir $@)
	$(HOST_CXX) $(CXXFLAGS) $< $(LIB) -o $@ $(LDFLAGS)

verify-headers: $(THIS_DIR)/include/freesched.h $(THIS_DIR)/include/ggml_freesilicon.h $(THIS_DIR)/include/pocketcoder_runner.h
	$(HOST_CC) -x c -std=c11 -fsyntax-only -Wall -Wextra -Werror -I$(THIS_DIR)/include $(THIS_DIR)/include/freesched.h
	$(HOST_CXX) -x c++ -std=c++17 -fsyntax-only -Wall -Wextra -Werror -I$(THIS_DIR)/include $(THIS_DIR)/include/freesched.h
	$(HOST_CC) -x c -std=c11 -fsyntax-only -Wall -Wextra -Werror -I$(THIS_DIR)/include $(THIS_DIR)/include/ggml_freesilicon.h
	$(HOST_CXX) -x c++ -std=c++17 -fsyntax-only -Wall -Wextra -Werror -I$(THIS_DIR)/include $(THIS_DIR)/include/ggml_freesilicon.h
	$(HOST_CC) -x c -std=c11 -fsyntax-only -Wall -Wextra -Werror -I$(THIS_DIR)/include $(THIS_DIR)/include/pocketcoder_runner.h
	$(HOST_CXX) -x c++ -std=c++17 -fsyntax-only -Wall -Wextra -Werror -I$(THIS_DIR)/include $(THIS_DIR)/include/pocketcoder_runner.h

test: $(TEST_C11) $(TEST_CXX) $(TEST_PC) $(BENCH_BIN) $(SMOKE_BIN) $(PC_BIN) twin-test
	FREESCHED_MOCK=1 $(TEST_C11)
	FREESCHED_MOCK=1 $(TEST_CXX)
	FREESCHED_MOCK=1 $(TEST_PC)
	$(PC_BIN) --bench-synthetic --iters 20
	$(SMOKE_BIN) --profile laguna --json
	$(SMOKE_BIN) --profile malibu --mock --json

smoke: $(SMOKE_BIN)
	$(SMOKE_BIN) --profile laguna --json

twin-test: $(BENCH_BIN)
	python3 -m unittest discover -s $(THIS_DIR)/tests -p 'test_*.py' -v

bench: $(BENCH_BIN)
	FREESCHED_MOCK=1 $(BENCH_BIN)
	$(THIS_DIR)/twin/freesilicon-twin bench --freesched 2000

audit:
	$(THIS_DIR)/scripts/clean-room-audit.sh

verify: verify-headers test bench audit

clean:
	rm -rf $(BUILD_DIR)
