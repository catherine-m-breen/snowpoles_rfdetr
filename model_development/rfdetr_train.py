'''
This is a backup copy for what is running on slurm 
57413408
'''

import numpy as np
import supervision as sv
import torch
from PIL import Image

#from rfdetr import RFDETRMedium, RFDETRSegSmall, RFDETRSegMedium, RFDETRKeypointPreview, RFDETRSegLarge
#from rfdetr.util.coco_classes import COCO_CLASSES


## paths
# cmbreen@discover33:/discover/nobackup/cmbreen/rfdetr_snow> ls
# dataset_rfd_detr  rf-detr-keypoint-preview-xlarge.pth  rf-detr-seg-medium.pt

#dataset = '/discover/nobackup/cmbreen/rfdetr_snow/dataset_rfd_detr'
#model_path = '/discover/nobackup/cmbreen/rfdetr_snow/rf-detr-seg-medium.pt'
#outputs = '/discover/nobackup/cmbreen/rfdetr_snow/outputs'
#exp_name = 'SegMedium_bs8_gas2_e200'

#model = RFDETRSegMedium(resolution=624)
#model.train(dataset_dir=dataset.location, epochs=200, batch_size=8, grad_accum_steps=2)

# viz every 10 epochs on a 5 examples; same 5 examples everytime 
# add in early stopping 
# plot the train adn val loss and save
# save the model checkpoint 

import torch
import os
import cv2
import glob
import numpy as np
import supervision as sv
from PIL import Image

from rfdetr import RFDETRSegMedium, RFDETRSegNano, RFDETRSegSmall

# --- 1. Paths and Configuration ---
dataset = '/discover/nobackup/cmbreen/rfdetr_snow/dataset_rfd_detr'
model_path = '/discover/nobackup/cmbreen/rfdetr_snow/rf-detr-seg-medium.pt'
outputs = '/discover/nobackup/cmbreen/rfdetr_snow/outputs'
exp_name = 'SegNano_bs16_5percent_e10' # Updated to bs16 based on 448 res

os.makedirs(outputs, exist_ok=True)

# --- 4. Post-Training Visualization on 5 Examples ---
# Once training finishes, we load the BEST weights it just saved
best_model_path = os.path.join(outputs, exp_name, 'checkpoint_best_total.pth')

# --- 2. Initialize Model ---
# Ensure resolution matches your dataset!
model = RFDETRSegNano(resolution=448)

# load local weights because Discover compute nodes have no internet
#model.load(model_path) 

if os.path.exists(best_model_path):
    print(f"Model already trained! Found checkpoint at: {best_model_path}")
    print("Skipping training and moving straight to visualization...")
else:
    print("Starting training on Discover GPU...")
    model.train(
        dataset_dir=dataset,
        epochs=10,
        batch_size=16,          
        grad_accum_steps=2,
        patience=5,            
        project=outputs,        
        name=exp_name,          
        workers=16,             
        device='cuda',        
        fraction=0.05 ## increasing to 10%
    )
    # rfdetr quirk: it ignores 'name' and creates an 'output' folder. Let's rename it automatically.
    bad_output_dir = os.path.join(outputs, 'output')
    correct_output_dir = os.path.join(outputs, exp_name)
    
    if os.path.exists(bad_output_dir):
        # If the target directory already exists (e.g. from a crashed run), handle it to avoid an error
        if not os.path.exists(correct_output_dir):
            os.rename(bad_output_dir, correct_output_dir)
        else:
            print(f"Warning: {correct_output_dir} already exists. Files may need manual moving.")
            
    print(f"Training complete. Weights and loss charts are saved in {correct_output_dir}")


if os.path.exists(best_model_path):
    print(f"Loading best weights for inference from: {best_model_path}")
    print(best_model_path)
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
else:
    print("Could not find best.pt for visualization. Check training logs.")



    import torch

# 1. Load the checkpoint file into memory using PyTorch
checkpoint = torch.load(best_model_path, map_location='cpu')

# 2. Extract the actual weights (state_dict) from the file
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
elif 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
elif 'model' in checkpoint:
    state_dict = checkpoint['model']
else:
    state_dict = checkpoint 
    
# 3. Apply the weights to your model
try:
    model.load_state_dict(state_dict)
    print("Weights loaded successfully using model.load_state_dict()!")
except AttributeError:
    # If the outer wrapper doesn't accept it, apply it to the internal model
    model.model.load_state_dict(state_dict)
    print("Weights loaded successfully using model.model.load_state_dict()!")

# 4. Put the model in evaluation mode for predictions
if hasattr(model, 'eval'):
    model.eval()
elif hasattr(model.model, 'eval'):
    model.model.eval()