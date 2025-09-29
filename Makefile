GPU_SM_ARCH=
MAX_QUERY_LEN=
N_CODE=
N_PENALTY=

GPU_COMPUTE_ARCH=$(subst sm,compute,$(GPU_SM_ARCH))
NVCC=/apps/cuda/12.9.0/bin/nvcc
CC=g++

SRC_DIR=./src/
OBJ_DIR=./obj/
LIB_DIR=./lib/
INCLUDE_DIR=./include/

SOURCES= args_parser.cpp host_batch.cpp ctors.cpp interfaces.cpp res.cpp gasal_align.cu
LOBJS=$(patsubst %,%o,$(SOURCES))
LOBJS_PATH=$(addprefix $(OBJ_DIR),$(LOBJS))
VPATH=src:obj:lib

YELLOW=\033[1;33m
NC=\033[0m # No Color

# --- sanity checks -----------------------------------------------------------
ifeq ($(GPU_SM_ARCH),)
error1:
	@echo "Must specify GPU architecture as sm_xx (e.g., sm_86, sm_90)"
endif
ifeq ($(MAX_QUERY_LEN),)
error2:
	@echo "Must specify maximum sequence length (e.g., MAX_QUERY_LEN=4096)"
endif
ifeq ($(N_CODE),)
error3:
	@echo "Must specify the code for 'N' (e.g., N_CODE=0x4E for ASCII 'N')"
endif

# --- compiler flags (add PIC everywhere) ------------------------------------
NVCC_CMN = -c -g -O3 -std=c++11 -Xptxas -Werror --default-stream per-thread \
           --gpu-architecture=$(GPU_COMPUTE_ARCH) --gpu-code=$(GPU_SM_ARCH) \
           -lineinfo --ptxas-options=-v -Xcompiler -Wall,-fPIC
CC_CMN   = -c -g -O3 -std=c++11 -Wall -fPIC

# Preprocessor defines
DEFS_BASE = -DMAX_QUERY_LEN=$(MAX_QUERY_LEN) -DN_CODE=$(N_CODE)
ifdef N_PENALTY
DEFS = $(DEFS_BASE) -DN_PENALTY=$(N_PENALTY)
else
DEFS = $(DEFS_BASE)
endif

# --- pattern rules -----------------------------------------------------------
%.cuo: %.cu
	$(NVCC) $(NVCC_CMN) $(DEFS) $< -o $(OBJ_DIR)$@

%.cppo: %.cpp
	$(CC) $(CC_CMN) $(DEFS) -Werror $< -o $(OBJ_DIR)$@

# --- default targets ---------------------------------------------------------
all: clean makedir libgasal.a libgasal.so

makedir:
	@mkdir -p $(OBJ_DIR) $(LIB_DIR) $(INCLUDE_DIR)
	@cp $(SRC_DIR)/*.h $(INCLUDE_DIR)
	@sed -i "s/MAX_QUERY_LEN=[0-9]\{1,9\}/MAX_QUERY_LEN=$(MAX_QUERY_LEN)/" ./test_prog/Makefile

libgasal.a: $(LOBJS)
	ar -csru $(LIB_DIR)$@ $(LOBJS_PATH)
ifndef N_PENALTY
	@echo ""
	@echo -e "${YELLOW}WARNING:${NC} N_PENALTY is not defined"
endif

# Build shared library from the same PIC objects
libgasal.so: $(LOBJS)
	$(NVCC) -shared -o $(LIB_DIR)/$@ $(LOBJS_PATH) -lcudart

clean:
	rm -f -r $(OBJ_DIR) $(LIB_DIR) $(INCLUDE_DIR) *~ *.exe *.cppo *.cuo *.txt *~

# headers for dependency hint
gasal_align.cuo: gasal.h gasal_kernels.h
