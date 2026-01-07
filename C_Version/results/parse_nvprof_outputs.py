import os
import re
import csv

# Target metrics and valid kernel names
METRICS = [
    "achieved_occupancy",
    "ipc",
    "sm_efficiency",
    "dram_read_throughput",
    "stall_memory_dependency",
    "flop_count_sp",
    "dram_utilization",
    "gld_efficiency",
]
KERNEL_NAMES = [
    "svm_inference_threaded_kernel_2d",
    "svm_inference_threaded_kernel_general",
    "svm_inference_batched_kernel",
    "svm_inference_kernel",
    "svm_inference_batched_kernel",
    "svm_inference_thread_parallel_kernel"
]

output_dirs = ["outputs1", "outputs2"]

def extract_basic_kernel_time(lines):
    for line in lines:
        for kernel in KERNEL_NAMES:
            if kernel in line:
                parts = line.strip().split()
                for part in parts:
                    if "ms" in part:
                        try:
                            val = float(part.replace("ms", "")) * 1000
                            return kernel, val
                        except ValueError:
                            continue
                    elif "us" in part:
                        try:
                            val = float(part.replace("us", ""))
                            return kernel, val
                        except ValueError:
                            continue
    return None, None

def extract_cpu_time_us(lines):
    patterns = [
        r"CPU BATCHED elapsed \(sec\):\s*([0-9.eE+-]+)",
        r"Batched CPU elapsed \(sec\):\s*([0-9.eE+-]+)"
    ]
    for line in lines:
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                try:
                    return float(match.group(1)) * 1e6
                except ValueError:
                    return None
    return None

def extract_metrics(lines):
    metrics = {m: None for m in METRICS}
    in_kernel_block = False

    for line in lines:
        if any(kernel in line for kernel in KERNEL_NAMES):
            in_kernel_block = True
            continue

        if in_kernel_block:
            for metric in METRICS:
                if re.search(rf"\b{metric}\b", line):
                    avg_field = line.strip().split()[-1]
                    try:
                        if metric == "dram_utilization":
                            metrics[metric] = avg_field.strip()
                        elif "%" in avg_field:
                            metrics[metric] = float(avg_field.replace("%", ""))
                        elif "GB/s" in avg_field:
                            metrics[metric] = float(avg_field.replace("GB/s", ""))
                        else:
                            metrics[metric] = float(avg_field)
                    except ValueError:
                        metrics[metric] = None
    return metrics

def sort_key(row):
    # Extract test type (THREAD_PARALLEL, BLOCK_PARALLEL, etc.)
    filename = row["filename"]
    match = re.match(r"([A-Z_]+)", filename)
    test_type = match.group(1) if match else ""
    
    # Extract any numbers to sort numerically (e.g., DIM_2 => 2)
    numbers = list(map(int, re.findall(r"\d+", filename)))
    return (test_type, numbers)

# Parse each directory separately
for dir_name in output_dirs:
    parsed_data = []
    for filename in os.listdir(dir_name):
        if filename.endswith(".txt"):
            path = os.path.join(dir_name, filename)
            with open(path, "r") as f:
                lines = f.readlines()

            fixed_lines = []
            for line in lines:
                if "==== NVPROF BASIC ====" in line and not line.strip().endswith("===="):
                    parts = line.split("==== NVPROF BASIC ====", 1)
                    fixed_lines.append("==== NVPROF BASIC ====\n")
                    fixed_lines.append(parts[1] if len(parts) > 1 else "")
                elif "==== NVPROF METRICS ====" in line and not line.strip().endswith("===="):
                    parts = line.split("==== NVPROF METRICS ====", 1)
                    fixed_lines.append("==== NVPROF METRICS ====\n")
                    fixed_lines.append(parts[1] if len(parts) > 1 else "")
                else:
                    fixed_lines.append(line)

            try:
                basic_start = next(i for i, line in enumerate(fixed_lines) if "==== NVPROF BASIC ====" in line) + 1
                metrics_start = next(i for i, line in enumerate(fixed_lines) if "==== NVPROF METRICS ====" in line) + 1

                basic_lines = fixed_lines[basic_start:metrics_start-1]
                metrics_lines = fixed_lines[metrics_start:]

                kernel, kernel_time_us = extract_basic_kernel_time(basic_lines)
                cpu_time_us = extract_cpu_time_us(basic_lines)

                if not kernel:
                    continue

                metric_values = extract_metrics(metrics_lines)

                parsed_data.append({
                    "filename": filename,
                    "kernel": kernel,
                    "kernel_time_us": kernel_time_us,
                    "cpu_time_us": cpu_time_us,
                    **metric_values
                })
            except Exception as e:
                print(f"Error parsing {filename}: {e}")

    # Sort parsed data before writing
    parsed_data.sort(key=sort_key)

    # Write to CSV per directory
    output_csv = f"parsed_{dir_name}.csv"
    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = ["filename", "kernel", "kernel_time_us", "cpu_time_us"] + METRICS
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in parsed_data:
            writer.writerow(row)

    print(f"Parsed data saved to: {output_csv}")
