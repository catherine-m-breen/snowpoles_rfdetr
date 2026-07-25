'''
conda activate /Users/cmbreen/miniforge3/envs/rfdetr_snow
Best model: checkpoint_best_total.pt 

'''

import numpy as np
import supervision as sv
from PIL import Image
import os
import cv2
import glob
import numpy as np
import supervision as sv
from rfdetr import RFDETRSegNano
import matplotlib.pyplot as plt
import math
from pathlib import Path
import pandas as pd
import tqdm

print("--- Snowpole Setup ---")
'''
We need to know two things related to the metadata 
1. Camera Name 
2. Pixel to Centimeter Conversion 
3. We also need to know which image is the reference image. Usually this is image #1, but might change depending 
3. Anything else (saved as dictionary); things like lat/long, reference notes etc

We need to know the total height of the full pole in cm and the length of the full pole in centimeters
OR we need to know the length in pixels of a "10cm" top

either of these two values results in the "pixel to centimeter conversion" which is 

# of centimeters (either full pole or 10cm top) / # of pixels (either for full pole or 10cm top)
'''

print("-" * 20)
print("Welcome to the Snow Pole Labeling Script!! \n We will walk through a series of questions to determine how to do the labeling. \n If you are familiar with this, please see the README for more configuration instructions.")
print("-" * 20)

print("--- Snowpole Setup ---")
'''
We need to know two things related to the metadata 
1. Camera Name 
2. Pixel to Centimeter Conversion 
3. We also need to know which image is the reference image. Usually this is image #1, but might change depending 
4. Anything else (saved as dictionary); things like lat/long, reference notes etc
'''

saved_configs = input('Do you want to run with presaved configurations? Type "Y" if you have updated the code, type "N" if you are running this for the first time: ').strip().upper()

if saved_configs == 'Y': 
    camera_name = 'TLS_N4W'
    camera_season = '2016-2017'
    #camera_image_path = '/Volumes/MyBook/poles_dataset/native_res_snex17/TLS-N4W' # Example Windows path
    camera_image_path = '/Volumes/MyBook/snowpoles_dataset_master/raw_images/SnEx_Raleigh_2016-17/TLS-A1N'
    pole_type = 'Full Pole'
    total_pole_cm = 304.8 
    pixel_centimeter_conversion = 0.264 
    ref_img_path_actual = 9 
    other_info = 'First 9 photos are foggy'
    location_information ='(39.032, -108.216)'
    viz_out_dir = '/Users/cmbreen/Documents/snowpoles/cosgrove/rfdetr_outputs'
    snow_depth_output = '/Users/cmbreen/Documents/snowpoles/cosgrove/rfdetr_snowdepth'
    print("-" * 20)
    print("Configurations Loaded Automatically")
    print(f"  Camera Name:          {camera_name}")
    print(f"  Season:               {camera_season}")
    print(f"  Image Path:           {camera_image_path}")
    print(f"  Pole Type:            {pole_type}")
    print(f"  Total Pole (cm):      {total_pole_cm}")
    print(f"  Pixel/cm Conversion:  {pixel_centimeter_conversion}")
    print(f"  Reference Image Idx:  {ref_img_path_actual}")
    print(f"  Location Info:        {location_information}")
    print(f"  Other Info:           {other_info}")
    print(f"  Viz Output Dir:       {viz_out_dir}")
    print(f"  Data Output Dir:      {snow_depth_output}")
    print("-" * 20)

##### Start and metadata ####### 
if saved_configs == "N": 
    camera_name = input("Enter the camera id, such as CameraA or TLS-A1N or Site1, etc (no quotes needed): ")
    camera_season = input("Enter the water year (e.g., 2019-2020, 2021-2022 etc): ")
    camera_image_path = input(r'Enter the full camera path from your computer. Make sure to enter the full path such as /Users/Documents/[Camera_Folder] on Mac or C:\Users\Documents\Camera_Folder on Windows ').strip('"').strip("'")
    
    #pole_info = input("Do you know the length of the full pole or does this pole have a 10 centimeter top? Type Y for Yes and N for No.")
    #pole_type = input("What is the length of the full pole in centimeters? (Put NA if non-applicable): ")
    total_pole_cm_input = input("What is the length of the full pole in centimeters? (Put NA if non-applicable): ")
    if total_pole_cm_input != 'NA':
        total_pole_cm = float(total_pole_cm_input)
    else:
        total_pole_cm = None
    
    pixel_centimeter_conversion = input("What is the pixel_centimeter_conversion? put NA if unsure: ")
    if pixel_centimeter_conversion != 'NA':
        pixel_centimeter_conversion = float(pixel_centimeter_conversion)
    
    ref_img_is_first = input("Is the first image the correct reference image? (Y for yes N for No): ").strip().upper()
    if ref_img_is_first == 'N':
        ref_img_path_actual = int(input("Enter the image number to use as reference (e.g., 9 for the 9th image): "))
    else:
        ref_img_path_actual = 1
    
    print("-" * 20)
    other_info = input("Is there any other information to store for this camera? For example, information related to malfunction etc. \n Put NA if nothing: ")
    location_information = input("Enter location information (long, lat) format, if don't know just put NA: ")
    #### output information ####
    viz_out_dir = Path(input(r"Enter visualization output folder path: ").strip('"').strip("'"))
    snow_depth_output = Path(input(r"Enter snow depth output folder path: ").strip('"').strip("'"))
    print("-" * 20)
    print("Configurations for Script")
    print("-" * 20)


#############

############# open camera folder ##########################
images = sorted(glob.glob(os.path.join(Path(camera_image_path), '*.[jJ][pP]*[gG]')))
if len(images) == 0:
    print(f"WARNING: No images found in {camera_image_path}! Check that the path is correct.")
else:
    # Set the exact path to the reference image so later code can use it
    ref_img_path = images[ref_img_path_actual - 1]
##############################################################


####### first get the pixel to centimeter conversion ###### 
## pull the 9th image or whatever image is lisetd for this:  ref_img_path_actual 
if pixel_centimeter_conversion == 'NA': 
    print("-" * 20)
    print('No pixel centimeter conversion factor stored. Loading the reference/ snow-free image to identify the pixel-centimeter conversion')
    print("-" * 20)
    calibration_target = float(input("Enter the known length in cm you will click (e.g., total pole length or 10): "))


    # Read and convert image for matplotlib
    ref_image_cv = cv2.imread(Path(ref_img_path))
    ref_image_rgb = cv2.cvtColor(ref_image_cv, cv2.COLOR_BGR2RGB)
    
    print("\n*** INSTRUCTIONS ***")
    print("A window will open. Click exactly TWO points: the TOP and BOTTOM of your calibration target.")
    print("Middle click or Right click to undo a point if you mess up.")

     
    # Plot the reference image and use ginput
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(ref_image_rgb)
    ax.set_title("Click top and bottom of the calibration target")
    
    # ginput(2) waits for 2 clicks, timeout=0 means it waits forever
    points = plt.ginput(2, timeout=0) 
    plt.close()
    
    if len(points) == 2:
        x1, y1 = points[0]
        x2, y2 = points[1]
        
        # Calculate euclidean distance in pixels
        pixel_distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        pixel_centimeter_conversion =  calibration_target / pixel_distance
        print("-" * 20)
        print('Pixel centimeter conversion saved!')
        print(f'Length in pixels: {pixel_distance:.2f}')
        print(f'Conversion factor (cm/px): {pixel_centimeter_conversion:.5f}')
        print("-" * 20)
    else:
        print("Error: Did not get 2 points. You'll need to restart the script or hardcode the conversion.")
        pixel_centimeter_conversion = 1.0 # fallback to prevent crash

##############


################# now run the model #############

######## download the model ##########

best_model_path = '/Users/cmbreen/code/snowpoles_rfdetr/checkpoint_best_total.pth'
# Removed the trailing /* here so os.path.join below works properly

os.makedirs(viz_out_dir, exist_ok=True)

color = sv.ColorPalette.from_hex([
    "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", "#b266ff",
    "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
])

if os.path.exists(best_model_path):
    print(f"Loading best weights for inference from: {best_model_path}")
    model = RFDETRSegNano(pretrain_weights=best_model_path, resolution=480)
    
    # --- CALCULATE CONVERSION FACTOR FROM REFERENCE IMAGE ---
    print(f"\nAnalyzing bare ground reference image: {ref_img_path}")
    ref_image = cv2.imread(ref_img_path)
    if ref_image is None:
        raise ValueError("Could not read the reference image. Please check the path.")
        
    ref_detections = model.predict(ref_image)
    
    if len(ref_detections.xyxy) == 0:
        raise ValueError("No pole detected in the reference image!")
        
    # Assuming the first detected object is the pole we want
    x_min_ref, y_min_ref, x_max_ref, y_max_ref = ref_detections.xyxy[0]
    bare_pole_px = y_max_ref - y_min_ref
    
    # Conversion factor: cm per pixel
    print(f'{total_pole_cm},{bare_pole_px}')
    conversion_factor = total_pole_cm / bare_pole_px
    print(f"-> Reference Pole length in pixels: {bare_pole_px:.2f} px")
    print(f"-> Calculated Conversion Factor: {conversion_factor:.4f} cm/px")
    print("-" * 20)
    
    # --- RUN PREDICTIONS ON SNOWY IMAGES ---
    images = glob.glob(os.path.join(camera_image_path, '*.[jJ][pP]*[gG]'))
    sample_images = images #[:5] 
    
    print("\nGenerating predictions for 5 test samples...")
    csv_file_path = os.path.join(snow_depth_output, f"{camera_name}_snowdepth.csv")
    results_data = [] 
    
    for i, img_path in tqdm.tqdm(enumerate(sample_images)):
        base_name = os.path.basename(img_path)
        print(f"\nProcessing {base_name}...")
        image = cv2.imread(img_path)
        
        detections = model.predict(image)
        
        # Loop through all detected objects in this image
        for j, xyxy in enumerate(detections.xyxy):
            x_min, y_min, x_max, y_max = xyxy
            pole_length_px = y_max - y_min
            
            # Apply conversion factor to get visible length in cm
            visible_length_cm = pole_length_px * conversion_factor
            
            # Snow depth is total height minus what is currently visible
            snow_depth_cm = total_pole_cm - visible_length_cm
            
            print(f"  Pole Detection {j+1}:")
            print(f"    - Length in pixels: {pole_length_px:.2f} px")
            print(f"    - Visible length: {visible_length_cm:.2f} cm")
            print(f"    - Estimated Snow Depth: {snow_depth_cm:.2f} cm")
            results_data.append({
                'camera_id': camera_name,
                'filename': base_name,
                'snowdepth': snow_depth_cm,
                'pixellength': pole_length_px,
                'conversion': conversion_factor
            })
        
        
        # Annotate
        if i % 20: ## save every 20 for examples  
            h, w = image.shape[:2]
            thickness = sv.calculate_optimal_line_thickness(resolution_wh=(w, h))
            color_annotator = sv.ColorAnnotator(color=color)
            polygon_annotator = sv.PolygonAnnotator(color=color, thickness=thickness)
            
            annotated_image = image.copy()
            annotated_image = color_annotator.annotate(scene=annotated_image, detections=detections)
            annotated_image = polygon_annotator.annotate(scene=annotated_image, detections=detections)
            
            # Convert OpenCV BGR image to PIL RGB Image for thumbnailing/saving
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(annotated_image_rgb)

            # Create thumbnail and save
            pil_img.thumbnail((800, 800))
        
            out_name = f"pred_{i+1}_{base_name}"
            save_path = os.path.join(viz_out_dir, out_name)

            pil_img.save(save_path)
            print(f"  -> Saved visualization: {out_name}")

    df = pd.DataFrame(results_data)
    df.to_csv(csv_file_path, index=False)
    
    print(f"\nAll visualizations saved to: {viz_out_dir}")
    print(f"Data saved to CSV using pandas: {csv_file_path}")
    print(f"\nAll visualizations saved to: {viz_out_dir}")
else:
    print(f"Error: Model not found at {best_model_path}")