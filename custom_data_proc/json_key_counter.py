import json

def count_keys_in_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    return len(data.keys())

file_path = "scene_gt.json" 
key_count = count_keys_in_json(file_path)
print(f"JSON {key_count} key。")
