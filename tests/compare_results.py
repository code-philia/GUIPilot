import csv
import os
import sys


def load_results(file_path):
    results = {}
    if not os.path.exists(file_path):
        return results
    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["id"], row["mutation"], row["matcher"], row["checker"])
            results[key] = {
                "tp": int(row["tp"]),
                "fp": int(row["fp"]),
                "fn": int(row["fn"]),
            }
    return results


def calculate_metrics(data):
    tp = data["tp"]
    fp = data["fp"]
    fn = data["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_results.py")
        print("       <base_results.csv> <head_results.csv>")
        sys.exit(1)

    base_file = sys.argv[1]
    head_file = sys.argv[2]
    base_results = load_results(base_file)
    head_results = load_results(head_file)

    if not base_results or not head_results:
        print("Error: One or both result files are empty or missing.")
        sys.exit(1)

    regression = False
    threshold = 0.05

    for key, head_data in head_results.items():
        if key not in base_results:
            continue
        base_data = base_results[key]
        base_p, base_r = calculate_metrics(base_data)
        head_p, head_r = calculate_metrics(head_data)

        if head_p < base_p - threshold:
            print("  !!! Precision drop detected")
            regression = True
        if head_r < base_r - threshold:
            print("  !!! Recall drop detected")
            regression = True

    if regression:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
