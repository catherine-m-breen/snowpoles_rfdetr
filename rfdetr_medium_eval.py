import os
import cv2
import glob
import numpy as np
import supervision as sv
from PIL import Image

from rfdetr import RFDETRSegMedium, RFDETRSegNano

# --- 1. Paths and Configuration ---
dataset = '/discover/nobackup/cmbreen/rfdetr_snow/dataset_rfd_detr'
model_path = '/discover/nobackup/cmbreen/rfdetr_snow/rf-detr-seg-nano.pt'
outputs = '/discover/nobackup/cmbreen/rfdetr_snow/outputs/nano_outputs'
# exp_name = 'SegMedium_bs16_gas2_e200' # Updated to bs16 based on 448 res

os.makedirs(outputs, exist_ok=True)

# --- 2. Initialize Model ---
# Ensure resolution matches your dataset!
model = RFDETRSegNano(resolution=448)

# --- 4. Post-Training Visualization on 5 Examples ---
# Once training finishes, we load the BEST weights it just saved
best_model_path = '/discover/nobackup/cmbreen/rfdetr_snow/output/checkpoint_best_total.pth'
if os.path.exists(best_model_path):
    print(f"Loading best weights for inference from: {best_model_path}")
    model.load(best_model_path)
    
    # Grab 5 random images from the test set
    test_images_dir = os.path.join(dataset, 'test')
    test_images = glob.glob(os.path.join(test_images_dir, '*.[jJ][pP]*[gG]'))
    sample_images = test_images[:5] # Grab the first 5
    
    viz_out_dir = os.path.join(outputs, exp_name, 'custom_viz_samples')
    os.makedirs(viz_out_dir, exist_ok=True)
    
    polygon_annotator = sv.PolygonAnnotator()
    keypoint_annotator = sv.KeyPointAnnotator() # Assuming sv has keypoint annotator available
    
    print("Generating predictions for 5 test samples...")
    for img_path in sample_images:
        img_name = os.path.basename(img_path)
        image = cv2.imread(img_path)
        
        # Run prediction
        result = model.predict(image)
        
        # Annotate
        annotated_image = polygon_annotator.annotate(scene=image, detections=result)
        # Note: Depending on your supervision version, keypoint annotation might differ slightly
        # annotated_image = keypoint_annotator.annotate(scene=annotated_image, keypoints=result.keypoints)
        
        # Save
        out_path = os.path.join(viz_out_dir, f"pred_{img_name}")
        cv2.imwrite(out_path, annotated_image)
        
    print(f"Visualizations saved to: {viz_out_dir}")
