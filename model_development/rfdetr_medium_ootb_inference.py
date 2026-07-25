import numpy as np
import supervision as sv

from PIL import Image

from rfdetr import RFDETRSegSmall, RFDETRSegMedium, RFDETRSegNano
from rfdetr.util.coco_classes import COCO_CLASSES
import os
import random

# # --- 1. Paths and Configuration ---
# dataset = '/discover/nobackup/cmbreen/rfdetr_snow/dataset_rfd_detr'
# outputs = '/discover/nobackup/cmbreen/rfdetr_snow/ootb_outputs'
# #exp_name = 'SegMedium_bs16_gas2_e200' # Updated to bs16 based on 448 res
# os.makedirs(outputs, exist_ok=True)

# # --- 2. Initialize Model ---
# # Ensure resolution matches your dataset!
# model1 = RFDETRSegNano(resolution=448)
# model2 = RFDETRSegSmall(resolution=448)
# model3 = RFDETRSegMedium(resolution=448)

# # High-contrast color palette for clear group differentiation
# color = sv.ColorPalette.from_hex([
#     "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", "#b266ff",
#     "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
# ])


# def ootb_model(dataset, model,model_type, output_path):

#     # --- 3. Gather Dataset Images ---
#     # Find all common image formats in the dataset path (including subdirectories)
#     valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
#     all_image_paths = []

#     for root, dirs, files in os.walk(dataset):
#         for file in files:
#             if file.endswith(valid_extensions):
#                 all_image_paths.append(os.path.join(root, file))


#     # --- 4. Random Sample & Evaluate ---
#     # Pick 10 random images (or fewer if the dataset is smaller than 10)
#     num_samples = min(10, len(all_image_paths))
#     sampled_paths = random.sample(all_image_paths, num_samples)

#     print(f"Found {len(all_image_paths)} images. Evaluating on {num_samples} random samples...")

#     for i, img_path in enumerate(sampled_paths):
#         print(f"Processing [{i+1}/{num_samples}]: {os.path.basename(img_path)}")
        
#         try:
#             # Load and resize
#             image = Image.open(img_path).convert("RGB") # ensure it's RGB
#             # Predict
#             detections = model.predict(image, threshold=0.1)

#             # Annotate
#             thickness = sv.calculate_optimal_line_thickness(resolution_wh=image.size)
#             color_annotator = sv.ColorAnnotator(color=color)
#             polygon_annotator = sv.PolygonAnnotator(color=color, thickness=thickness)
            
#             annotated_image = image.copy()
#             annotated_image = color_annotator.annotate(scene=annotated_image, detections=detections)
#             annotated_image = polygon_annotator.annotate(scene=annotated_image, detections=detections)
            
#             annotated_image.thumbnail((800, 800))
#             annotated_image
#             # Save out to output directory
#             base_name = os.path.basename(img_path)
#             out_name = f"pred_{i+1}_{base_name}"
#             save_path = os.path.join(output_path, model_type, out_name)
#             os.makedirs(save_path, exist_ok=True)

#             annotated_image.save(save_path)
            
#         except Exception as e:
#             print(f"Failed to process {img_path}. Error: {e}")

#     print(f"Done! Annotated images saved to: {outputs}")
#     return

# ootb_model(dataset, model1, 'nano', outputs)
# ootb_model(dataset, model2, 'small', outputs)
# ootb_model(dataset, model3, 'medium', outputs)


import numpy as np
import supervision as sv
from PIL import Image
from rfdetr import RFDETRSegSmall, RFDETRSegMedium, RFDETRSegNano
import os
import random

# --- 1. Paths and Configuration ---
dataset = '/discover/nobackup/cmbreen/rfdetr_snow/dataset_rfd_detr'
outputs = '/discover/nobackup/cmbreen/rfdetr_snow/ootb_outputs'
os.makedirs(outputs, exist_ok=True)

# --- 2. Gather Dataset Images & Sample ONCE ---
valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
all_image_paths = []

for root, dirs, files in os.walk(dataset):
    for file in files:
        if file.endswith(valid_extensions):
            all_image_paths.append(os.path.join(root, file))

num_samples = min(10, len(all_image_paths))
# We sample here so all 3 models evaluate the EXACT same 10 images
sampled_paths = random.sample(all_image_paths, num_samples)

print(f"Found {len(all_image_paths)} images. Evaluating on {num_samples} shared random samples...")

# --- 3. Initialize Models ---
# Ensure resolution matches your dataset!
model1 = RFDETRSegNano(resolution=448)
model2 = RFDETRSegSmall(resolution=448)
model3 = RFDETRSegMedium(resolution=448)

# High-contrast color palette
color = sv.ColorPalette.from_hex([
    "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", "#b266ff",
    "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
])

def ootb_model(image_paths, model, model_type, output_path):
    # Create the directory for this specific model type
    model_out_dir = os.path.join(output_path, model_type)
    os.makedirs(model_out_dir, exist_ok=True)
    
    print(f"\n--- Running evaluation for {model_type.upper()} ---")
    
    for i, img_path in enumerate(image_paths):
        print(f"Processing [{i+1}/{len(image_paths)}]: {os.path.basename(img_path)}")
        
        try:
            # Load image
            image = Image.open(img_path).convert("RGB")
            
            # Predict
            detections = model.predict(image, threshold=0.1)

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
            save_path = os.path.join(model_out_dir, out_name)

            annotated_image.save(save_path)
            
        except Exception as e:
            print(f"Failed to process {img_path}. Error: {e}")

    print(f"Done! Annotated images for {model_type} saved to: {model_out_dir}")

# --- 4. Run Evaluation ---
ootb_model(sampled_paths, model1, 'nano', outputs)
ootb_model(sampled_paths, model2, 'small', outputs)
ootb_model(sampled_paths, model3, 'medium', outputs)