import numpy as np
import supervision as sv

from PIL import Image

from rfdetr import RFDETRMedium, RFDETRSegSmall, RFDETRSegMedium, RFDETRKeypointPreview, RFDETRSegLarge
from rfdetr.util.coco_classes import COCO_CLASSES


## paths
# cmbreen@discover33:/discover/nobackup/cmbreen/rfdetr_snow> ls
# dataset_rfd_detr  rf-detr-keypoint-preview-xlarge.pth  rf-detr-seg-medium.pt

# dataset = '/discover/nobackup/cmbreen/rfdetr_snow/dataset_rfd_detr'
# model_path = '/discover/nobackup/cmbreen/rfdetr_snow/rf-detr-seg-medium.pt'
# outputs = '/discover/nobackup/cmbreen/rfdetr_snow/outputs'
# exp_name = 'SegMedium_bs8_gas2_e200'

# model = RFDETRSegMedium(resolution=624)
# model.train(dataset_dir=dataset.location, epochs=200, batch_size=8, grad_accum_steps=2)

# viz every 10 epochs on a 5 examples; same 5 examples everytime 
# add in early stopping 
# plot the train adn val loss and save
# save the model checkpoint 

import os
import cv2
import glob
import numpy as np
import supervision as sv
from PIL import Image

from rfdetr import RFDETRSegMedium

# --- 1. Paths and Configuration ---
dataset = '/discover/nobackup/cmbreen/rfdetr_snow/dataset_rfd_detr'
model_path = '/discover/nobackup/cmbreen/rfdetr_snow/rf-detr-seg-medium.pt'
outputs = '/discover/nobackup/cmbreen/rfdetr_snow/outputs'
exp_name = 'SegMedium_bs16_gas2_e200' # Updated to bs16 based on 448 res

os.makedirs(outputs, exist_ok=True)

# --- 2. Initialize Model ---
# Ensure resolution matches your dataset!
model = RFDETRSegMedium(resolution=448)

# load local weights because Discover compute nodes have no internet
# model.load(model_path) 

# --- 3. Train Model ---
print("Starting training on Discover GPU...")
model.train(
    dataset_dir=dataset,
    epochs=10,
    batch_size=32,          # 448x448 uses less memory, batch size 16 should easily fit
    grad_accum_steps=2,
    patience=5,            # EARLY STOPPING: Stop if no improvement in 20 epochs
    project=outputs,        # Save output charts/weights here
    name=exp_name,          # Folder name for this specific experiment
    workers=16,              # HPC optimization: speeds up image loading to feed GPU
    device='cuda',         # Explicitly target the first GPU
    fraction = 0.01
)
print(f"Training complete. Weights and loss charts are saved in {outputs}/{exp_name}")

#model.train(dataset_dir=dataset.location, epochs=10, batch_size=8, grad_accum_steps=2)