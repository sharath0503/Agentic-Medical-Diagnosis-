import os
import json

def fetch_reports():
    fallback_file = os.path.join("data", "diagnosis_reports.json")
    print("Reading local diagnosis reports...")
    if os.path.exists(fallback_file):
        print(f"Reading local reports file: {fallback_file}")
        with open(fallback_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        if len(lines) == 0:
            print("Local reports file is empty.")
            return
        print(f"\nFound {len(lines)} reports:\n")
        for idx, line in enumerate(lines, start=1):
            print(f"Report #{idx}")
            try:
                report_data = json.loads(line)
                print(json.dumps(report_data, indent=4))
            except json.JSONDecodeError:
                print("  [INVALID JSON LINE]")
                print(line)
            print("-" * 50)
    else:
        print("No local reports file found.")

if __name__ == "__main__":
    fetch_reports()
