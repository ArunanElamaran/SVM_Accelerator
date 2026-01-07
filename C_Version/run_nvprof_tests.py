import os
import subprocess
from pathlib import Path

# Define parameters to test
RUN_MODES = ["THREAD_PARALLEL", "BLOCK_PARALLEL"]
DIM_VALUES = [2, 8, 32]
THREADS_MAX_VALUES = [4, 8, 32, 64]
NUM_INPUTS_VALUES = [30, 1000, 100000, 10000000]

# Default values
DEFAULT_DIM = 2
DEFAULT_THREADS_MAX = 32
DEFAULT_NUM_INPUTS = 10000000
DEFAULT_RUN_MODE = "THREAD_PARALLEL"

# Ensure output directories exist
results_dir_svm1 = Path("outputs1")
results_dir_svm1.mkdir(exist_ok=True)

results_dir_svm2 = Path("outputs2")
results_dir_svm2.mkdir(exist_ok=True)

def run_command(cmd):
    """Run shell command and return combined stdout+stderr output as string."""
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return result.stdout.decode()

def make_svm1(run_mode, dim, threads_max, num_inputs):
    """Build svm1 with custom macro flags."""
    make_cmd = (
        f"make clean && "
        f"make svm1 RUN_MODE={run_mode} DIM={dim} THREADS_MAX={threads_max} NUM_INPUTS={num_inputs}"
    )
    return run_command(make_cmd)

def profile_and_store(run_mode, param_name, param_value):
    """Run nvprof commands and save their outputs."""
    output_filename = f"{run_mode}_{param_name}_{param_value}.txt"
    output_path = results_dir_svm1 / output_filename

    print(f"Running test: {output_filename}")

    cmd1 = "./svm1"
    nvprof1 = f"nvprof {cmd1}"
    nvprof2 = (
        f"nvprof -m achieved_occupancy,ipc,sm_efficiency,"
        f"dram_read_throughput,stall_memory_dependency,"
        f"flop_count_sp,dram_utilization,gld_efficiency {cmd1}"
    )

    # Run and collect outputs
    out1 = run_command(nvprof1)
    out2 = run_command(nvprof2)

    # Write to file
    with open(output_path, "w") as f:
        f.write(f"==== NVPROF BASIC ===={out1}\n")
        f.write(f"==== NVPROF METRICS ===={out2}\n")

# for run_mode in RUN_MODES:
#     # Vary DIM
#     for dim in DIM_VALUES:
#         make_svm1(run_mode, dim, DEFAULT_THREADS_MAX, DEFAULT_NUM_INPUTS)
#         profile_and_store(run_mode, "DIM", dim)

#     # Vary THREADS_MAX
#     for tmax in THREADS_MAX_VALUES:
#         make_svm1(run_mode, DEFAULT_DIM, tmax, DEFAULT_NUM_INPUTS)
#         profile_and_store(run_mode, "THREADS_MAX", tmax)

#     # Vary NUM_INPUTS
#     for n in NUM_INPUTS_VALUES:
#         make_svm1(run_mode, DEFAULT_DIM, DEFAULT_THREADS_MAX, n)
#         profile_and_store(run_mode, "NUM_INPUTS", n)

# Additional SVM2 profiling automation

# Define constants for SVM2 testing
SVM_GPU_MODES = ["THREAD", "BLOCK"]
SVM_NUM_SV_VALUES = [16, 32]
BLOCK_SIZE_VALUES = [32, 64, 128, 256]
THREAD_BLOCK_SIZE_VALUES = [32, 64, 128, 256]

DEFAULT_SVM_NUM_SV = 16
DEFAULT_BLOCK_SIZE = 256
DEFAULT_THREAD_BLOCK_SIZE = 128
DEFAULT_SVM_DIM = 2

def make_svm2(gpu_mode, svm_num_sv, block_size, thread_block_size):
    """Build svm2 with custom macro flags."""
    make_cmd = (
        f"make clean && "
        f"make svm2 SVM_GPU_MODE={gpu_mode} "
        f"SVM_NUM_SV={svm_num_sv} SVM_DIM={DEFAULT_SVM_DIM} "
        f"BLOCK_SIZE={block_size} THREAD_BLOCK_SIZE={thread_block_size}"
    )
    return run_command(make_cmd)

def profile_and_store_svm2(gpu_mode, param_name, param_value):
    """Run nvprof on svm2 and save outputs."""
    output_filename = f"svm2_{gpu_mode}_{param_name}_{param_value}.txt"
    output_path = results_dir_svm2 / output_filename

    print(f"Running SVM2 test: {output_filename}")

    cmd1 = "./svm2"
    nvprof1 = f"nvprof {cmd1}"
    nvprof2 = (
        f"nvprof -m achieved_occupancy,ipc,sm_efficiency,"
        f"dram_read_throughput,stall_memory_dependency,"
        f"flop_count_sp,dram_utilization,gld_efficiency {cmd1}"
    )

    out1 = run_command(nvprof1)
    out2 = run_command(nvprof2)

    with open(output_path, "w") as f:
        f.write(f"==== NVPROF BASIC ===={out1}\n")
        f.write(f"==== NVPROF METRICS ===={out2}\n")

for mode in SVM_GPU_MODES:
    if mode == "THREAD":
        for val in SVM_NUM_SV_VALUES:
            make_svm2(mode, val, DEFAULT_BLOCK_SIZE, DEFAULT_THREAD_BLOCK_SIZE)
            profile_and_store_svm2(mode, "SVM_NUM_SV", val)

        for val in THREAD_BLOCK_SIZE_VALUES:
            make_svm2(mode, DEFAULT_SVM_NUM_SV, DEFAULT_BLOCK_SIZE, val)
            profile_and_store_svm2(mode, "THREAD_BLOCK_SIZE", val)

    elif mode == "BLOCK":
        for val in SVM_NUM_SV_VALUES:
            make_svm2(mode, val, DEFAULT_BLOCK_SIZE, DEFAULT_THREAD_BLOCK_SIZE)
            profile_and_store_svm2(mode, "SVM_NUM_SV", val)

        for val in BLOCK_SIZE_VALUES:
            make_svm2(mode, DEFAULT_SVM_NUM_SV, val, DEFAULT_THREAD_BLOCK_SIZE)
            profile_and_store_svm2(mode, "BLOCK_SIZE", val)
