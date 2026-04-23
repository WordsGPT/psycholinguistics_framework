import json

# Update these variables to your problem:
file1_path = 'iconicity_english/batchesA/batches.jsonl'
file2_path = 'iconicity_english/results_downloaded_from_openAI/batch_69e753cd16ac81908d76613e21e009c7_error.jsonl'
output_path = 'iconicity_english/batchesA/new_batches.jsonl'
# END: Update these variables to your problem:

# Step 1: Collect custom_id values from file2
custom_ids_file2 = set()

with open(file2_path, 'r', encoding='utf-8') as f2:
    for line in f2:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            if "custom_id" in data:
                custom_ids_file2.add(data["custom_id"])
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON line in file2: {line}")

# Step 2: Filter file1 and write results
with open(file1_path, 'r', encoding='utf-8') as f1, \
        open(output_path, 'w', encoding='utf-8') as out:
    
    for line in f1:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            if data.get("custom_id") in custom_ids_file2:
                out.write(json.dumps(data) + "\n")
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON line in file1: {line}")

