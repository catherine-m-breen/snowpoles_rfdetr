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
viz_out_dir = '/discover/nobackup/cmbreen/rfdetr_snow/outputs/nano_outputs'
# exp_name = 'SegMedium_bs16_gas2_e200' # Updated to bs16 based on 448 res

os.makedirs(viz_out_dir, exist_ok=True)

# --- 2. Initialize Model ---
# Ensure resolution matches your dataset!
#model = RFDETRSegNano(resolution=448)

color = sv.ColorPalette.from_hex([
    "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", "#b266ff",
    "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
])

# --- 4. Post-Training Visualization on 5 Examples ---
# Once training finishes, we load the BEST weights it just saved
best_model_path = '/discover/nobackup/cmbreen/rfdetr_snow/output/checkpoint_best_total.pth'
if os.path.exists(best_model_path):
    print(f"Loading best weights for inference from: {best_model_path}")
    model = RFDETRSegNano(pretrain_weights=best_model_path, resolution=448)
    
    # Grab 5 random images from the test set
    test_images_dir = os.path.join(dataset, 'test')
    test_images = glob.glob(os.path.join(test_images_dir, '*.[jJ][pP]*[gG]'))
    sample_images = test_images[:5] # Grab the first 5
    
    os.makedirs(viz_out_dir, exist_ok=True)
    
    # polygon_annotator = sv.PolygonAnnotator()
    # keypoint_annotator = sv.KeyPointAnnotator() # Assuming sv has keypoint annotator available
    
    print("Generating predictions for 5 test samples...")
    for img_path in sample_images:
        img_name = os.path.basename(img_path)
        image = cv2.imread(img_path)
        
        # Run prediction
        detections = model.predict(image)
        
       # Annotate
        thickness = sv.calculate_optimal_line_thickness(resolution_wh=image.size)
        color_annotator = sv.ColorAnnotator(color=color)
        polygon_annotator = sv.PolygonAnnotator(color=color, thickness=thickness)
        
        annotated_image = image.copy()
        annotated_image = color_annotator.annotate(scene=annotated_image, detections=detections)
        annotated_image = polygon_annotator.annotate(scene=annotated_image, detections=detections)
        
        # Create thumbnail and save
        annotated_image.thumbnail((800, 800))
        
        base_name = os.path.basename(img_path)
        out_name = f"pred_{i+1}_{base_name}"
        save_path = os.path.join(viz_out_dir, out_name)

        annotated_image.save(save_path)
        
    print(f"Visualizations saved to: {viz_out_dir}")
