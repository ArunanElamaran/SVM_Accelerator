CPP=g++
OPT?=-O3
CFLAGS= --std=c++11 -g -ggdb -gdwarf-3 $(OPT) -fsanitize=address
MODULE          := svm1 svm2

# === User-Tunable Parameters (Used in SVM1) ===
DIM := 2
NUM_INPUTS := 10000000

# For SVM1
RUN_MODE := THREAD_PARALLEL
THREADS_MAX := 32

# For SVM2
SVM_NUM_SV := 16
SVM_GPU_MODE := THREAD
BLOCK_SIZE := 256
THREAD_BLOCK_SIZE := 128


# === One-Line Macro Flags for SVM1 & SVM2 ===
MACROS1 := -DRUN_MODE=$(RUN_MODE) -DDIM=$(DIM) -DTHREADS_MAX=$(THREADS_MAX) -DNUM_INPUTS=$(NUM_INPUTS)
MACROS2 := -DSVM_DIM=$(DIM) -DSVM_NUM_SV=$(SVM_NUM_SV) -DSVM_GPU_MODE=$(SVM_GPU_MODE) -DBLOCK_SIZE=$(BLOCK_SIZE) -DTHREAD_BLOCK_SIZE=$(THREAD_BLOCK_SIZE)

.PHONY: all clean

all: $(MODULE)

HEADERS = svm_linear.hpp

# Compile svm1 with macros
svm1: svm_linear.cu
	nvcc $^ -o $@ -lcudnn $(MACROS1)

# Compile svm2 without macros
svm2: svm.cu
	nvcc $^ -o $@ -lcudnn $(MACROS2)

clean:
	rm -f $(MODULE)