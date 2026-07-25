# '''

# Once all the images are converted to COCO format this script saves them to a dataset_rfd_detr folder that separates the images based off 
# network 

# '''

import os
import json
import shutil
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm

# # --- Configuration ---
# csv_path = '/Volumes/MyBook/snowpoles_dataset_master/labels_merged_allsites.csv'
# master_json_path = '/Volumes/MyBook/snowpoles_dataset_master/images_448_matched_COCO/annotations.json'
# master_images_dir = '/Volumes/MyBook/snowpoles_dataset_master/images_448_matched_COCO/images'
# output_dataset_dir = '/Volumes/MyBook/snowpoles_dataset_master/dataset_rfd_detr'

# # Move viz folder outside the train/val/test folders
# output_viz_dir = os.path.join(output_dataset_dir, 'viz_samples')

# # Lowercased for guaranteed matching
# SPLIT_CONFIG = {
#     'train': ['snowex20', 'uwcascades'],
#     'valid': ['snowex23'],
#     'test':  ['snowex17']
# }

# # --- 1. Load the CSV and clean strings ---
# print("Reading CSV to extract metadata...")
# df = pd.read_csv(csv_path)

# df['filename'] = df['filename'].astype(str).str.strip()
# df['dataset'] = df['dataset'].astype(str).str.strip().str.lower()
# df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

# df_unique = df.drop_duplicates(subset=['filename'])

# filename_to_dataset = df_unique.set_index('filename')['dataset'].to_dict()
# filename_to_datetime = df_unique.set_index('filename')['datetime'].to_dict()

# # --- 2. Load the master COCO JSON ---
# print("Loading master COCO JSON...")
# with open(master_json_path, 'r') as f:
#     master_data = json.load(f)

# images = master_data['images']
# annotations = master_data['annotations']

# # --- Robust Filename Matcher for CSV Lookup ---
# csv_filenames = sorted(list(filename_to_dataset.keys()), key=len, reverse=True)
# json_to_csv_name = {}

# for img in images:
#     json_fname = img['file_name']
#     for csv_fname in csv_filenames:
#         if json_fname.endswith(csv_fname):
#             json_to_csv_name[json_fname] = csv_fname
#             break

# # --- 3. Sort images into splits ---
# splits = {'train': [], 'valid': [], 'test': []}
# unassigned_images = []

# dataset_to_split = {}
# for split_name, dataset_list in SPLIT_CONFIG.items():
#     for ds_name in dataset_list:
#         dataset_to_split[ds_name] = split_name

# for img in images:
#     original_filename = json_to_csv_name.get(img['file_name'])
    
#     if original_filename:
#         dataset_category = filename_to_dataset.get(original_filename)
#         target_split = dataset_to_split.get(dataset_category)
        
#         if target_split:
#             splits[target_split].append(img)
#         else:
#             unassigned_images.append(img['file_name'])
#     else:
#         unassigned_images.append(img['file_name'])

# if unassigned_images:
#     print(f"\nWarning: {len(unassigned_images)} images skipped (dataset not in SPLIT_CONFIG or missing from CSV).")

# summary_stats = {}

# # --- 4. Function to create splits and visualizations ---
# def create_split(split_name, split_images):
#     if not split_images:
#         print(f"\nSkipping '{split_name}' (0 images assigned).")
#         return
        
#     print(f"\nProcessing '{split_name}' split...")
    
#     split_dir = os.path.join(output_dataset_dir, split_name)
#     viz_dir = os.path.join(output_viz_dir, split_name) 
#     os.makedirs(split_dir, exist_ok=True)
#     os.makedirs(viz_dir, exist_ok=True)
    
#     camera_set = set()
#     image_metadata_list = []
    
#     for img in split_images:
#         orig_name = json_to_csv_name.get(img['file_name'])
        
#         # --- NEW: Extract Camera ID directly from filename string ---
#         # e.g., "013098_TLS-N4W_WSCT2468.JPG" -> splits into ['013098', 'TLS-N4W', 'WSCT2468.JPG']
#         parts = img['file_name'].split('_')
#         if len(parts) >= 3:
#             cam = parts[1]  # The camera ID is the second chunk
#         else:
#             cam = "Unknown"
            
#         dt = filename_to_datetime.get(orig_name, pd.NaT)
#         camera_set.add(cam)
        
#         image_metadata_list.append({'img_info': img, 'camera': cam, 'datetime': dt})
        
#     print(f"  -> {len(split_images)} Images across {len(camera_set)} unique camera sites.")
#     summary_stats[split_name] = {'images': len(split_images), 'cameras': len(camera_set)}
    
#     split_image_ids = set(img['id'] for img in split_images)
#     split_annotations = [ann for ann in annotations if ann['image_id'] in split_image_ids]
#     ann_by_image = {ann['image_id']: ann for ann in split_annotations}
    
#     split_coco = {
#         "info": master_data.get("info", {}),
#         "categories": master_data["categories"],
#         "images": split_images,
#         "annotations": split_annotations
#     }
    
#     with open(os.path.join(split_dir, '_annotations.coco.json'), 'w') as f:
#         json.dump(split_coco, f)
        
#     # --- Generate 5 Diverse Visualizations ---
#     image_metadata_list = [item for item in image_metadata_list if not pd.isna(item['datetime'])]
#     image_metadata_list.sort(key=lambda x: x['datetime'])
    
#     selected_for_viz = []
#     seen_cameras = set()
    
#     if len(image_metadata_list) > 0:
#         chunk_size = max(1, len(image_metadata_list) // 5)
#         for i in range(0, len(image_metadata_list), chunk_size):
#             for item in image_metadata_list[i:]:
#                 if item['camera'] not in seen_cameras:
#                     selected_for_viz.append(item)
#                     seen_cameras.add(item['camera'])
#                     break
#             if len(selected_for_viz) == 5:
#                 break
                
#     print("  -> Generating visual samples...")
#     for idx, item in enumerate(selected_for_viz):
#         img_info = item['img_info']
#         ann = ann_by_image.get(img_info['id'])
#         if not ann: continue
            
#         src_path = os.path.join(master_images_dir, img_info['file_name'])
#         if not os.path.exists(src_path): continue
            
#         img = cv2.imread(src_path)
#         overlay = img.copy()
        
#         if ann.get('segmentation') and len(ann['segmentation'][0]) > 0:
#             poly = ann['segmentation'][0]
#             pts = np.array(poly).reshape((-1, 1, 2)).astype(np.int32)
#             cv2.fillPoly(overlay, [pts], (0, 255, 0))
            
#         kpts = ann.get('keypoints', [])
#         if len(kpts) >= 6:
#             cv2.circle(overlay, (int(kpts[0]), int(kpts[1])), 4, (0, 0, 255), -1)
#             cv2.circle(overlay, (int(kpts[3]), int(kpts[4])), 4, (0, 0, 255), -1)
            
#         blended = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)
#         dt_str = item['datetime'].strftime('%Y-%m-%d')
#         cv2.putText(blended, f"{item['camera']} | {dt_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
#         cv2.imwrite(os.path.join(viz_dir, f"sample_{idx+1}_{img_info['file_name']}"), blended)
        
#     # --- Copy the images ---
#     for img_info in tqdm(split_images, desc=f"  -> Copying {split_name} images"):
#         src_path = os.path.join(master_images_dir, img_info['file_name'])
#         dst_path = os.path.join(split_dir, img_info['file_name'])
#         if not os.path.exists(dst_path):
#             shutil.copy2(src_path, dst_path)

# # --- 5. Run the splits ---
# create_split('train', splits['train'])
# create_split('valid', splits['valid'])
# create_split('test', splits['test'])

# # --- 6. Final Summary ---
# print("\n" + "="*40)
# print("              SUMMARY")
# print("="*40)
# print(f"Output Directory: {output_dataset_dir}\n")
# for split in ['train', 'valid', 'test']:
#     if split in summary_stats:
#         imgs = summary_stats[split]['images']
#         cams = summary_stats[split]['cameras']
#         print(f"{split.upper():<7}: {imgs:>5} images across {cams:>3} camera sites")
# print("="*40)

import os
import json
import shutil
import pandas as pd
import cv2
import numpy as np
import random
import textwrap
from tqdm import tqdm

# --- Configuration ---
csv_path = '/Volumes/MyBook/snowpoles_dataset_master/labels_merged_allsites.csv'
master_json_path = '/Volumes/MyBook/snowpoles_dataset_master/images_448_matched_COCO/annotations.json'
master_images_dir = '/Volumes/MyBook/snowpoles_dataset_master/images_448_matched_COCO/images'
output_dataset_dir = '/Volumes/MyBook/snowpoles_dataset_master/dataset_rfd_detr'

# Move viz folder outside the train/val/test folders
output_viz_dir = os.path.join(output_dataset_dir, 'viz_samples')

# Lowercased for guaranteed matching
SPLIT_CONFIG = {
    'train': ['snowex20', 'uwcascades'],
    'valid': ['snowex23'],
    'test':  ['snowex17']
}

# --- 1. Load the CSV and clean strings ---
print("Reading CSV to extract metadata...")
df = pd.read_csv(csv_path)

df['filename'] = df['filename'].astype(str).str.strip()
df['dataset'] = df['dataset'].astype(str).str.strip().str.lower()

df_unique = df.drop_duplicates(subset=['filename'])
filename_to_dataset = df_unique.set_index('filename')['dataset'].to_dict()

# --- 2. Load the master COCO JSON ---
print("Loading master COCO JSON...")
with open(master_json_path, 'r') as f:
    master_data = json.load(f)

images = master_data['images']
annotations = master_data['annotations']

# --- Robust Filename Matcher for CSV Lookup ---
csv_filenames = sorted(list(filename_to_dataset.keys()), key=len, reverse=True)
json_to_csv_name = {}

for img in images:
    json_fname = img['file_name']
    for csv_fname in csv_filenames:
        if json_fname.endswith(csv_fname):
            json_to_csv_name[json_fname] = csv_fname
            break

# --- 3. Sort images into splits ---
splits = {'train': [], 'valid': [], 'test': []}
unassigned_images = []

dataset_to_split = {}
for split_name, dataset_list in SPLIT_CONFIG.items():
    for ds_name in dataset_list:
        dataset_to_split[ds_name] = split_name

for img in images:
    original_filename = json_to_csv_name.get(img['file_name'])
    
    if original_filename:
        dataset_category = filename_to_dataset.get(original_filename)
        target_split = dataset_to_split.get(dataset_category)
        
        if target_split:
            splits[target_split].append(img)
        else:
            unassigned_images.append(img['file_name'])
    else:
        unassigned_images.append(img['file_name'])

if unassigned_images:
    print(f"\nWarning: {len(unassigned_images)} images skipped (dataset not in SPLIT_CONFIG or missing from CSV).")

summary_stats = {}

# --- 4. Function to create splits and visualizations ---
def create_split(split_name, split_images):
    if not split_images:
        print(f"\nSkipping '{split_name}' (0 images assigned).")
        return
        
    print(f"\nProcessing '{split_name}' split...")
    
    split_dir = os.path.join(output_dataset_dir, split_name)
    viz_dir = os.path.join(output_viz_dir, split_name) 
    os.makedirs(split_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    
    # Dictionary to group images by camera: {'Cam1': [img1, img2], 'Cam2': [img3]}
    camera_groups = {}
    
    for img in split_images:
        # Extract Camera ID directly from filename string
        parts = img['file_name'].split('_')
        if len(parts) >= 3:
            cam = parts[1]  
        else:
            cam = "Unknown"
            
        if cam not in camera_groups:
            camera_groups[cam] = []
        camera_groups[cam].append(img)
        
    camera_list = list(camera_groups.keys())
    print(f"  -> {len(split_images)} Images across {len(camera_list)} unique camera sites.")
    
    summary_stats[split_name] = {
        'images': len(split_images), 
        'camera_count': len(camera_list),
        'camera_list': sorted(camera_list)
    }
    
    split_image_ids = set(img['id'] for img in split_images)
    split_annotations = [ann for ann in annotations if ann['image_id'] in split_image_ids]
    ann_by_image = {ann['image_id']: ann for ann in split_annotations}
    
    split_coco = {
        "info": master_data.get("info", {}),
        "categories": master_data["categories"],
        "images": split_images,
        "annotations": split_annotations
    }
    
    with open(os.path.join(split_dir, '_annotations.coco.json'), 'w') as f:
        json.dump(split_coco, f)
        
    # --- Generate Random Visualizations (5 different cameras) ---
    selected_for_viz = []
    random.shuffle(camera_list) # Shuffle to get different cameras
    
    # Pick up to 5 cameras, and grab 1 random image from each
    cameras_to_sample = camera_list[:min(5, len(camera_list))]
    
    for cam in cameras_to_sample:
        random_img = random.choice(camera_groups[cam])
        selected_for_viz.append((cam, random_img))
        
    # If there are fewer than 5 cameras, fill the rest with random images from the whole split
    while len(selected_for_viz) < 5 and len(split_images) > len(selected_for_viz):
        random_img = random.choice(split_images)
        # Avoid duplicates
        if not any(img['id'] == random_img['id'] for _, img in selected_for_viz):
            parts = random_img['file_name'].split('_')
            cam = parts[1] if len(parts) >= 3 else "Unknown"
            selected_for_viz.append((cam, random_img))
                
    print(f"  -> Generating {len(selected_for_viz)} visual samples...")
    for idx, (cam_name, img_info) in enumerate(selected_for_viz):
        ann = ann_by_image.get(img_info['id'])
        if not ann: continue
            
        src_path = os.path.join(master_images_dir, img_info['file_name'])
        if not os.path.exists(src_path): continue
            
        img = cv2.imread(src_path)
        overlay = img.copy()
        
        if ann.get('segmentation') and len(ann['segmentation'][0]) > 0:
            poly = ann['segmentation'][0]
            pts = np.array(poly).reshape((-1, 1, 2)).astype(np.int32)
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            
        kpts = ann.get('keypoints', [])
        if len(kpts) >= 6:
            cv2.circle(overlay, (int(kpts[0]), int(kpts[1])), 4, (0, 0, 255), -1)
            cv2.circle(overlay, (int(kpts[3]), int(kpts[4])), 4, (0, 0, 255), -1)
            
        blended = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)
        cv2.putText(blended, f"Camera: {cam_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.imwrite(os.path.join(viz_dir, f"sample_{idx+1}_{img_info['file_name']}"), blended)
        
    # --- Copy the images ---
    for img_info in tqdm(split_images, desc=f"  -> Copying {split_name} images"):
        src_path = os.path.join(master_images_dir, img_info['file_name'])
        dst_path = os.path.join(split_dir, img_info['file_name'])
        if not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)

# --- 5. Run the splits ---
create_split('train', splits['train'])
create_split('valid', splits['valid'])
create_split('test', splits['test'])

# --- 6. Final Summary ---
print("\n" + "="*80)
print("                                 SUMMARY")
print("="*80)
print(f"Output Directory: {output_dataset_dir}\n")
for split in ['train', 'valid', 'test']:
    if split in summary_stats:
        imgs = summary_stats[split]['images']
        cams = summary_stats[split]['camera_count']
        cam_list = summary_stats[split]['camera_list']
        
        print(f"{split.upper():<7}: {imgs:>5} images across {cams:>3} camera sites")
        
        cam_str = ", ".join(cam_list)
        wrapped_cams = textwrap.fill(cam_str, width=80, initial_indent="         Sites: ", subsequent_indent="                ")
        print(wrapped_cams)
        print("-" * 80)