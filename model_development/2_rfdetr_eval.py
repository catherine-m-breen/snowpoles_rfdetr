import os
import json
import cv2
import random
import numpy as np
import supervision as sv
from tqdm import tqdm
from PIL import Image

from rfdetr import RFDETRSegNano

# --- 1. Paths and Configuration ---
test_dir = '/discover/nobackup/cmbreen/rfdetr_snow/dataset_rfd_detr/test'
json_path = os.path.join(test_dir, '_annotations.coco.json')
weights_path = '/discover/nobackup/cmbreen/rfdetr_snow/outputs/SegNano_bs16_5percent_e10/checkpoint_best_total.pth'
eval_out_dir = '/discover/nobackup/cmbreen/rfdetr_snow/outputs/eval_side_by_side'

os.makedirs(eval_out_dir, exist_ok=True)

# High-contrast color palette for predictions
color = sv.ColorPalette.from_hex([
    "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", "#b266ff"
])

# --- 2. Initialize Fine-Tuned Model ---
print(f"Loading model weights from {weights_path}...")
model = RFDETRSegNano(pretrain_weights=weights_path, resolution=448)

# --- 3. Load COCO Annotations (Ground Truth) ---
print(f"Loading annotations from: {json_path}")
with open(json_path, 'r') as f:
    coco_data = json.load(f)

images = coco_data.get('images', [])
annotations = coco_data.get('annotations', [])

# Map image_id to its annotations for easy lookup
img_to_anns = {img['id']: [] for img in images}
for ann in annotations:
    img_to_anns[ann['image_id']].append(ann)

# --- 4. Sample 5% of the Test Dataset ---
fraction = 0.05
num_samples = max(1, int(len(images) * fraction))
sampled_images = random.sample(images, num_samples)

print(f"Evaluating on {num_samples} random images (approx {fraction*100}% of test set)...")

# --- 5. Generate Side-by-Side Visuals ---
for img_info in tqdm(sampled_images, desc="Generating Eval Images"):
    file_name = img_info.get('file_name')
    img_id = img_info.get('id')
    img_path = os.path.join(test_dir, file_name)
    
    if not os.path.exists(img_path):
        continue
        
    # Read base image
    original_img = cv2.imread(img_path)
    if original_img is None:
        continue
        
    height, width = original_img.shape[:2]
    
    # ---------------------------------------------------------
    # A. Create Ground Truth Image (Left Side)
    # ---------------------------------------------------------
    gt_img = original_img.copy()
    overlay = gt_img.copy()
    
    for ann in img_to_anns[img_id]:
        # Draw GT Polygons
        if ann.get('segmentation') and len(ann['segmentation']) > 0 and len(ann['segmentation'][0]) > 0:
            poly = ann['segmentation'][0]
            pts = np.array(poly).reshape((-1, 1, 2)).astype(np.int32)
            cv2.fillPoly(overlay, [pts], (0, 255, 0)) # Green mask
            
        # Draw GT Keypoints
        kpts = ann.get('keypoints', [])
        if len(kpts) >= 6: 
            x1, y1 = int(kpts[0]), int(kpts[1])
            x2, y2 = int(kpts[3]), int(kpts[4])
            cv2.circle(overlay, (x1, y1), 6, (0, 0, 255), -1) # Red dots
            cv2.circle(overlay, (x2, y2), 6, (0, 0, 255), -1)
            
    gt_img = cv2.addWeighted(overlay, 0.5, gt_img, 0.5, 0)
    cv2.putText(gt_img, "GROUND TRUTH", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # ---------------------------------------------------------
    # B. Create Model Prediction Image (Right Side)
    # ---------------------------------------------------------
    pred_img = original_img.copy()
    
    try:
        detections = model.predict(pred_img)
        
        thickness = sv.calculate_optimal_line_thickness(resolution_wh=(width, height))
        color_annotator = sv.ColorAnnotator(color=color)
        polygon_annotator = sv.PolygonAnnotator(color=color, thickness=thickness)
        
        pred_img = color_annotator.annotate(scene=pred_img, detections=detections)
        pred_img = polygon_annotator.annotate(scene=pred_img, detections=detections)
    except Exception as e:
        print(f"Prediction failed on {file_name}: {e}")
        
    cv2.putText(pred_img, "MODEL PREDICTION", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

    # ---------------------------------------------------------
    # C. Combine and Save
    # ---------------------------------------------------------
    # Stack the two images horizontally side-by-side
    combined_img = np.hstack((gt_img, pred_img))
    
    # Optional: Resize if the combined image is too massive to view easily
    # combined_img = cv2.resize(combined_img, (1600, int(1600 * (height / (width * 2)))))
    
    out_path = os.path.join(eval_out_dir, f"compare_{file_name}")
    cv2.imwrite(out_path, combined_img)

print(f"\nDone! Side-by-side evaluation images saved to: {eval_out_dir}")