
# evaluate_pipeline.py
# End-to-End Evaluation Script for Drone Traffic Project

import json

def calculate_count_accuracy(gt, pred):
    return (min(gt, pred) / max(gt, pred)) * 100 if max(gt, pred) > 0 else 0

def calculate_error(gt, pred):
    return abs(gt - pred)

def main():
    print("=== Drone Pipeline Evaluation ===")

    # ---- INPUT GROUND TRUTH ----
    gt_total = int(input("Enter Ground Truth Total Vehicles: "))
    gt_line = int(input("Enter Ground Truth Vehicles Passed Line: "))
    gt_speed = float(input("Enter Average Ground Truth Speed (km/h): "))

    # ---- INPUT MODEL OUTPUT ----
    pred_total = int(input("Enter Predicted Total Vehicles: "))
    pred_line = int(input("Enter Predicted Vehicles Passed Line: "))
    pred_speed = float(input("Enter Predicted Avg Speed (km/h): "))

    # ---- CALCULATIONS ----
    total_acc = calculate_count_accuracy(gt_total, pred_total)
    line_acc = calculate_count_accuracy(gt_line, pred_line)
    speed_error = calculate_error(gt_speed, pred_speed)

    results = {
        "Total Count Accuracy (%)": round(total_acc, 2),
        "Line Count Accuracy (%)": round(line_acc, 2),
        "Speed Error (km/h)": round(speed_error, 2)
    }

    # ---- SAVE RESULTS ----
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\n=== Results ===")
    for k, v in results.items():
        print(f"{k}: {v}")

    print("\nSaved to evaluation_results.json")

if __name__ == "__main__":
    main()
