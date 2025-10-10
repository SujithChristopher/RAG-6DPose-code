import os
import pickle
import json
import numpy as np
from PIL import Image

# Directories for PKL and JSON files
pkl_folder = "/home/bop/capture16_objects/gt_masks/object"
json_folder = "/home/bop/capture16_objects5/gt_pose"
output_image_folder = "/home/bop/h/train_real/000003/mask_visib"
output_json_file = "/home/bop/h/train_real/000003/scene_gt.json"

# Create output folder if it doesn't exist
os.makedirs(output_image_folder, exist_ok=True)

# Object to obj_id mapping
obj_to_id = {
    "bowl": 1,
    "cup": 2,
}

# Final result JSON
result_json = {}

# Processing files
pkl_files = sorted([f for f in os.listdir(pkl_folder) if f.endswith(".pkl")])
json_files = sorted([f for f in os.listdir(json_folder) if f.endswith(".json")])

for pkl_file, json_file in zip(pkl_files, json_files):
    pkl_path = os.path.join(pkl_folder, pkl_file)
    json_path = os.path.join(json_folder, json_file)
    base_name = os.path.splitext(pkl_file)[0]

    # Load PKL and JSON
    with open(pkl_path, "rb") as f:
        mask_data = pickle.load(f)
    with open(json_path, "r") as f:
        json_data = json.load(f)

    # Result for this PKL file
    result_json_entry = []
    
    msk_idx=0
    for obj_name, matrix in json_data.items():
        # Extract rotation and translation
        matrix = np.array(matrix)
        cam_R_m2c = matrix[:3, :3].flatten().tolist()
        cam_t_m2c = (matrix[:3, 3] * 1000).tolist()  # Convert to mm
        
        # Get obj_id
        obj_id = obj_to_id[obj_name]
        
        # Add to result JSON entry
        result_json_entry.append({
            "cam_R_m2c": cam_R_m2c,
            "cam_t_m2c": cam_t_m2c,
            "obj_id": obj_id
        })
        
        # Save mask images
        if obj_name in mask_data:
            mask = mask_data[obj_name]
            image_name = f"{base_name}_{msk_idx:06d}.png"
            image_path = os.path.join(output_image_folder, image_name)
            mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
            mask_image.save(image_path)
            msk_idx+=1
    # Add this PKL entry to the main JSON
    result_json[str(int(base_name))] = result_json_entry

# Save final JSON
with open(output_json_file, "w") as f:
    json.dump(result_json, f, indent=4)

print("Processing complete. Images and JSON generated.")
