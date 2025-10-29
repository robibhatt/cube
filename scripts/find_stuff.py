import os
import json


def find_all_jsons_with_few_ancestors(directory):
    matches = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if not filename.endswith(".json"):
                continue
            filepath = os.path.join(root, filename)
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)

                if "ancestors" in data and isinstance(data["ancestors"], list):
                    if len(data["ancestors"]) < 4:
                        matches.append(filepath)
                        print(filepath)

            except Exception as e:
                print(f"Error reading {filepath}: {e}")

    if matches:
        print("Found matching JSON files:")
        for path in matches:
            print(path)
    else:
        print("No matching JSON found.")

    return matches

if __name__ == '__main__':
    find_all_jsons_with_few_ancestors('experiment_runs/dkwl/learning_rate_l1_tuning')
# result = find_first_json_with_few_ancestors("/path/to/json/directory")
