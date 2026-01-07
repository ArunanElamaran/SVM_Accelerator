import csv
import re
from collections import defaultdict

csv_files = ["parsed_outputs1.csv", "parsed_outputs2.csv"]

def extract_test_category(filename):
    # Take the first three parts to group by something like THREAD_PARALLEL_DIM
    parts = filename.split('_')
    return '_'.join(parts[:3]) if len(parts) >= 3 else filename

def extract_numeric_suffix(filename):
    # Try to find common test pattern suffixes
    for tag in ["DIM_", "NUM_INPUTS_", "THREADS_MAX_"]:
        match = re.search(rf"{tag}(\d+)", filename)
        if match:
            return int(match.group(1))
    return float('inf')  # fallback to ensure these go last if no match

for csv_file in csv_files:
    print(f"\nSpeedups from: {csv_file}")
    categorized = defaultdict(list)

    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                kernel_time = float(row.get("kernel_time_us") or row.get("gpu_time_us") or 0)
                cpu_time = float(row["cpu_time_us"])
                if kernel_time > 0:
                    speedup = cpu_time / kernel_time
                    category = extract_test_category(row['filename'])
                    categorized[category].append((row['filename'], speedup))
            except (ValueError, ZeroDivisionError, KeyError):
                continue

    for category in sorted(categorized):
        print(f"\n[{category}]")
        sorted_items = sorted(
            categorized[category],
            key=lambda x: extract_numeric_suffix(x[0])
        )
        for fname, sp in sorted_items:
            print(f"{fname}: Speedup = {sp:.2f}x")
