import os
import json
import cv2
import numpy as np
from tqdm import tqdm

# --- Configuration ---
test_dir = '/Volumes/MyBook/snowpoles_dataset_master/dataset_rfd_detr/test'
json_path = os.path.join(test_dir, '_annotations.coco.json')
inspection_dir = '/Volumes/MyBook/snowpoles_dataset_master/dataset_rfd_detr/test_inspection'

# Create the output directory
os.makedirs(inspection_dir, exist_ok=True)

# --- 1. Load the Test COCO JSON ---
print(f"Loading annotations from: {json_path}")
with open(json_path, 'r') as f:
    coco_data = json.load(f)

images = coco_data.get('images', [])
annotations = coco_data.get('annotations', [])

print(f"Found {len(images)} images and {len(annotations)} annotations.")

# Create a lookup dictionary mapping image_id to the image info
image_lookup = {img['id']: img for img in images}

# --- 2. Generate Inspection Images ---
print(f"Saving superimposed images to: {inspection_dir}")

# We iterate through annotations (labels) rather than images
for ann in tqdm(annotations, desc="Generating inspection images"):
    image_id = ann.get('image_id')
    img_info = image_lookup.get(image_id)
    
    if not img_info:
        continue
        
    file_name = img_info.get('file_name')
    img_path = os.path.join(test_dir, file_name)
    
    if not os.path.exists(img_path):
        print(f"\nWarning: Image not found at {img_path}")
        continue
        
    # Read the image
    img = cv2.imread(img_path)
    if img is None:
        continue
        
    overlay = img.copy()
    
    # 1. Draw Polygon Mask
    if ann.get('segmentation') and len(ann['segmentation']) > 0 and len(ann['segmentation'][0]) > 0:
        poly = ann['segmentation'][0]
        # Reshape the flat list [x1,y1, x2,y2...] into pairs [[x1,y1], [x2,y2]...]
        pts = np.array(poly).reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(overlay, [pts], (0, 255, 0)) # Green mask
        
    # Blend to make the polygon semi-transparent
    blended = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)
    
    # 2. Draw Keypoints
    kpts = ann.get('keypoints', [])
    if len(kpts) >= 6: # COCO keypoints are [x, y, visibility, x, y, visibility]
        x1, y1 = int(kpts[0]), int(kpts[1])
        x2, y2 = int(kpts[3]), int(kpts[4])
        
        # Red dots for top and bottom
        cv2.circle(blended, (x1, y1), 4, (0, 0, 255), -1) 
        cv2.circle(blended, (x2, y2), 4, (0, 0, 255), -1)
        
    # 3. Add text overlay with filename so you can easily trace bad labels back to the CSV
    cv2.putText(blended, file_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # 4. Save the image
    out_path = os.path.join(inspection_dir, f"inspected_{file_name}")
    cv2.imwrite(out_path, blended)

print("\nDone! Check the 'test_inspection' folder to see what is going wrong.")