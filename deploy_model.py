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
import supervision as sv
from rfdetr import RFDETRSegNano
import matplotlib.pyplot as plt
import math
from pathlib import Path
import pandas as pd
import tqdm
import datetime
from matplotlib.widgets import Button, RectangleSelector# Added for the Reset button
#from matplotlib.widgets import Button, RectangleSelector 
## additional packages ## 
#import plotly.express as px ## need to do conda install
#import IPython
import re
import torch 
import gc
import json 

torch.set_num_threads(2) 

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

    # new crop feature in configs (in case picture is super noisy/ lots of poles and they don't tilt)
    apply_crop = False
    crop_coords = None

    ##### clicked center for anchor point #####
    clicked_x_center = None
    
    # Updated to create nested output structure
    base_output_dir = Path("outputs")
    camera_out_dir = base_output_dir / camera_name
    csv_dir = camera_out_dir / 'csv'
    viz_dir = camera_out_dir / 'sample_outputs'
    
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
    print(f"  Base Output Dir:      {base_output_dir}")
    print("-" * 20)

##### Start and metadata ####### 
if saved_configs == "N": 
    camera_name = input("Enter the camera id, such as CameraA or TLS-A1N or Site1, etc (no quotes needed): ")
    camera_season = input("Enter the water year (e.g., 2019-2020, 2021-2022 etc): ")
    raw_path = input(r'Enter the full camera path from your computer. Make sure to enter the full path such as /Users/Documents/[Camera_Folder] on Mac or C:\Users\Documents\Camera_Folder on Windows ').strip().strip('"').strip("'")
    # 2. Fix common Mac copy-paste errors (missing leading slash)
    if raw_path.startswith("Volumes/") or raw_path.startswith("Users/"):
        raw_path = "/" + raw_path
        
    # 3. Expand '~' if they used it (e.g., ~/Documents/...)
    camera_image_path = os.path.expanduser(raw_path)
    total_pole_cm_input = input("What is the length of the full pole in centimeters? (Put NA if non-applicable) \n If you do not know the height of the full pole, you WILL need a calibration section that you do know the length of (e.g., 10cm top, etc.): ")
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
    # Ask for base directory, default to "outputs" in the current folder if left blank
    base_out_input = input(r"Enter base output folder path (press Enter to just use 'outputs' in current folder): ").strip().strip('"').strip("'")
    base_output_dir = Path(base_out_input) if base_out_input else Path("outputs")
    
    # Automatically generate the nested folder structure
    camera_out_dir = base_output_dir / camera_name
    csv_dir = camera_out_dir / 'csv'
    viz_dir = camera_out_dir / 'sample_outputs'

    ########### new crop feature ###########
    ##### warning the model should track the pole, but if it's really not working you can crop the image (as long as you know the pole won't tilt a ton)
    ############# OPEN CAMERA FOLDER TO FIND REFERENCE IMAGE ##########################
    images = sorted(glob.glob(os.path.join(Path(camera_image_path), '*.[jJ][pP]*[gG]')))
    if len(images) == 0:
        print(f"WARNING: No images found in {camera_image_path}! Check that the path is correct.")
        exit()
    else:
        # Set the exact path to the reference image so later code can use it
        ref_img_path = images[ref_img_path_actual - 1]
    ###################################################################################

    if saved_configs == "N":
        # --- NEW: BOUNDING BOX / CROP SELECTION ---
        crop_state = {'coords': None}
        apply_crop_input = input("\nDo you want to apply a bounding box (crop) to the images? (Y/N): ").strip().upper()
        apply_crop = (apply_crop_input == 'Y')
        
        if apply_crop:
            print("\n*** CROP SELECTION ***")
            print("A window will open. Click and drag to draw a bounding box around the area of interest.")
            print("You can adjust the corners. Press 'Enter' when you are satisfied with the crop.")
            
            ref_image_cv_crop = cv2.imread(str(ref_img_path))
            ref_image_rgb_crop = cv2.cvtColor(ref_image_cv_crop, cv2.COLOR_BGR2RGB)
            
            fig_crop, ax_crop = plt.subplots(figsize=(10, 8))
            ax_crop.imshow(ref_image_rgb_crop)
            ax_crop.set_title("Draw Bounding Box (Click & Drag)\nPress 'Enter' when done")
            
            def line_select_callback(eclick, erelease):
                x1, y1 = eclick.xdata, eclick.ydata
                x2, y2 = erelease.xdata, erelease.ydata
                # Ensure coordinates are within image bounds and integers
                h, w = ref_image_rgb_crop.shape[:2]
                xmin = max(0, int(min(x1, x2)))
                ymin = max(0, int(min(y1, y2)))
                xmax = min(w, int(max(x1, x2)))
                ymax = min(h, int(max(y1, y2)))
                crop_state['coords'] = (xmin, ymin, xmax, ymax)
            
            rs = RectangleSelector(ax_crop, line_select_callback,
                                useblit=True,
                                button=[1],  # Left mouse button
                                minspanx=5, minspany=5,
                                spancoords='pixels',
                                interactive=True)
                                
            def on_key_crop(event):
                if event.key == 'enter':
                    plt.close(fig_crop)
                    
            fig_crop.canvas.mpl_connect('key_press_event', on_key_crop)
            plt.show()
            
            crop_coords = crop_state['coords']
            if crop_coords is None:
                print("Warning: No crop selected. Proceeding without crop.")
                apply_crop = False
            else:
                print(f"Crop selected (xmin, ymin, xmax, ymax): {crop_coords}")
        else:
            crop_coords = None
        # ------------------------------------------

    
    print("-" * 20)
    print("Configurations for Script")
    print(f"Camera ID:                   {camera_name}")
    print(f"Water Year:                  {camera_season}")
    print(f"Image Directory:             {camera_image_path}")
    print(f"Total Pole Length (cm):      {total_pole_cm_input}")
    print(f"Pixel/cm Conversion:         {pixel_centimeter_conversion}")
    print(f"Reference Image Number:      {ref_img_path_actual}")
    print(f"Location Info:               {location_information}")
    print(f"Other Info/Notes:            {other_info}")
    print(f"Base Output Directory:       {base_output_dir}")
    print(f"Camera Output Directory:     {camera_out_dir}")
    print(f"Apply Crop:                  {apply_crop}")
    print("-" * 20)


##########zoom function for calibration clicking ##########
def enable_scroll_zoom_and_pan(ax, base_scale=1.2):
    """Enables mouse-wheel zooming and right-click panning for a matplotlib axis"""
    pan_state = {'is_panning': False, 'start_x': None, 'start_y': None, 'start_xlim': None, 'start_ylim': None}

    def zoom(event):
        if event.inaxes != ax: return
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        xdata = event.xdata 
        ydata = event.ydata 
        
        if event.button == 'up':
            scale_factor = 1 / base_scale # zoom in
        elif event.button == 'down':
            scale_factor = base_scale     # zoom out
        else:
            return

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        relx = (cur_xlim[1] - xdata)/(cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])

        ax.set_xlim([xdata - new_width * (1-relx), xdata + new_width * (relx)])
        ax.set_ylim([ydata - new_height * (1-rely), ydata + new_height * (rely)])
        ax.figure.canvas.draw_idle()

    def press(event):
        # Button 3 is the RIGHT mouse button
        if event.button == 3 and event.inaxes == ax:
            pan_state['is_panning'] = True
            pan_state['start_x'] = event.x
            pan_state['start_y'] = event.y
            pan_state['start_xlim'] = ax.get_xlim()
            pan_state['start_ylim'] = ax.get_ylim()

    def release(event):
        if event.button == 3:
            pan_state['is_panning'] = False

    def motion(event):
        if pan_state['is_panning'] and pan_state['start_x'] is not None:
            dx_pixels = event.x - pan_state['start_x']
            dy_pixels = event.y - pan_state['start_y']
            bbox = ax.get_window_extent()
            dx_data = dx_pixels * (pan_state['start_xlim'][1] - pan_state['start_xlim'][0]) / bbox.width
            dy_data = dy_pixels * (pan_state['start_ylim'][1] - pan_state['start_ylim'][0]) / bbox.height
            
            ax.set_xlim(pan_state['start_xlim'][0] - dx_data, pan_state['start_xlim'][1] - dx_data)
            ax.set_ylim(pan_state['start_ylim'][0] - dy_data, pan_state['start_ylim'][1] - dy_data)
            ax.figure.canvas.draw_idle()

    ax.figure.canvas.mpl_connect('scroll_event', zoom)
    ax.figure.canvas.mpl_connect('button_press_event', press)
    ax.figure.canvas.mpl_connect('button_release_event', release)
    ax.figure.canvas.mpl_connect('motion_notify_event', motion)



'''
Download model

'''
######## download the model ##########

def download_models(): 
    '''
    see the Zenodo page for the latest models
    '''
    root =  os.getcwd()
    save_path = f"{root}"
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    url = 'https://zenodo.org/records/21570659/files/checkpoint_best_total.pth'
    
    # download if does not exist  
    if not (Path(save_path) / 'checkpoint_best_total.pth').exists():
    
        wget_command = f'wget {url} -P {save_path}'
        output_file = os.path.join(save_path, url.split("/")[-1]).replace("\\","/")
        curl_command = f'curl -L --ssl-no-revoke "{url}" -o "{output_file}"'
        print(curl_command)
        os.system(curl_command)
        return print('\n model downloaded! \n')
    else:
        return print('model already saved')

best_model_path = download_models()

# Create the nested directory structure automatically
os.makedirs(csv_dir, exist_ok=True)
os.makedirs(viz_dir, exist_ok=True)



#############

############# open camera folder ##########################
images = sorted(glob.glob(os.path.join(Path(camera_image_path), '*.[jJ][pP]*[gG]')))
if len(images) == 0:
    print(f"WARNING: No images found in {camera_image_path}! Check that the path is correct.")
    exit()
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

    print("\n*** INSTRUCTIONS ***")
    print("For this code to run, you need to tell the computer how many pixels in the image correspond to centimeters. This is because the computer first finds the length in pixels and then converts to centimeters. \n"
    "To do so, we pick an area of the image that we know the length of in cm and click the length (by clicking the top and bottom). The computer then figures out what this is in pixels, and then it figures out the pixel to \n"
    "centimeter conversion by dividing the pixel length by the centimeter length (pixel/ centimeter = pixel to centimeter conversion). Now when it calculates the snow depth in pixels, it can convert that to real-world centimeters. \n "
    "Usually your calibration target is the full pole (if you know the length in centimeters) or a 10-cm top or some other segment that you have measured in the field. This next step is very important.")

    calibration_target = float(input("\n Enter the known length in *cm* of the area you will click. This needs to be either the full pole height in cm if you have that measurement OR a small segment \n" \
    "that you know the height of, such as a 10 cm top: "))

    print("A window will open. Click exactly TWO points: the TOP and BOTTOM of your calibration target.")
    print("Middle click or Right click to undo a point if you mess up.")
    # Read and convert image for matplotlib (Convert Path to string to avoid cv2 errors)
    ref_image_cv = cv2.imread(str(ref_img_path))
    ref_image_rgb = cv2.cvtColor(ref_image_cv, cv2.COLOR_BGR2RGB)

     
    # Plot the reference image and use ginput
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(ref_image_rgb)
    ax.set_title("Click top and bottom of calibration target\n(Scroll = Zoom | Right-Click+Drag = Pan | 'Backspace' = Undo Point | 'r' = Reset View)")
    
    # Save the original boundaries for resetting
    orig_xlim = ax.get_xlim()
    orig_ylim = ax.get_ylim()

    def reset_view(event):
        ax.set_xlim(orig_xlim)
        ax.set_ylim(orig_ylim)
        fig.canvas.draw_idle()

    # Enable Reset via the 'r' keyboard shortcut
    fig.canvas.mpl_connect('key_press_event', lambda event: reset_view(event) if event.key == 'r' else None)

    # Enable Reset via UI Button
    ax_reset = plt.axes([0.4, 0.05, 0.2, 0.075])  # [left, bottom, width, height]
    btn_reset = Button(ax_reset, 'Reset Zoom')
    btn_reset.on_clicked(reset_view)
    
    # Attach your custom scroll and pan logic to the axes
    enable_scroll_zoom_and_pan(ax)

    
    # # ginput(2) waits for 2 clicks, timeout=0 means it waits forever
    # points = plt.ginput(2, timeout=0) 
    # plt.close()
    
        # --- CUSTOM GINPUT REPLACEMENT ---
    points = []
    markers = [] # Stores the red dots so we can remove them if you undo

    def on_click(event):
        # 1. Ignore clicks that are NOT on the main image (e.g., ignore the Reset Button)
        if event.inaxes != ax:
            return
            
        # 2. Only register LEFT clicks (button 1). Right click (3) is for panning.
        if event.button == 1:
            points.append((event.xdata, event.ydata))
            
            # Draw a visual red dot where the user clicked
            marker, = ax.plot(event.xdata, event.ydata, 'r+', markersize=6)
            markers.append(marker)
            fig.canvas.draw()
            
            # If we have 2 points, we are done!
            if len(points) == 2:
                plt.pause(0.2) # Give a tiny pause so they can see the second red dot
                plt.close(fig)

    def on_key(event):
        # Handle 'r' to reset zoom
        if event.key == 'r':
            reset_view()
            
        # Handle 'backspace' to undo the last point
        elif event.key == 'backspace' and len(points) > 0:
            points.pop()          # Remove the coordinate
            marker = markers.pop() # Remove the red dot
            marker.remove()
            fig.canvas.draw_idle()

    # Connect our custom functions to the figure
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Show the plot and block the script until the window is closed
    plt.show() 
    # --------------------------------- 

    if len(points) == 2:
        x1, y1 = points[0]
        x2, y2 = points[1]
        
        # Calculate euclidean distance in pixels
        pixel_distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        clicked_x_center = (x1 + x2) / 2.0 ### save this value so you know what the clicked center is 
        
        pixel_centimeter_conversion =  calibration_target / pixel_distance
        print("-" * 20)
        print('Pixel centimeter conversion saved!')
        print(f'Length in pixels: {pixel_distance:.2f}')
        print(f'Conversion factor (cm/px): {pixel_centimeter_conversion:.5f}')
        print("-" * 20)
    else:
        print("Error: Did not get 2 points. You'll need to restart the script or hardcode the conversion.")
        pixel_centimeter_conversion = 1.0 # fallback to prevent crash
    plt.close()
##############
if saved_configs == "N":
    config_dict = {
        "camera_name": camera_name,
        "camera_season": camera_season,
        "camera_image_path": str(camera_image_path),
        "pole_type": "Full Pole",
        "total_pole_cm": total_pole_cm,
        "pixel_centimeter_conversion": pixel_centimeter_conversion,
        "ref_img_path_actual": ref_img_path_actual,
        "other_info": other_info,
        "location_information": location_information,
        "base_output_dir": str(base_output_dir),
        "apply_crop": apply_crop,
        "crop_coords": crop_coords
    }
    
    # Create the config file path directly inside the camera's output directory
    config_file_path = camera_out_dir / 'snowpole_config.json'
    
    with open(config_file_path, 'w') as f:
        json.dump(config_dict, f, indent=4)
        
    print("-" * 20)
    print(f"Configuration saved to '{config_file_path}'!")
    print("-" * 20)
  

################# now run the model #############

######## download the model ##########

best_model_path = './checkpoint_best_total.pth'

# Create the nested directory structure automatically
os.makedirs(csv_dir, exist_ok=True)
os.makedirs(viz_dir, exist_ok=True)

color = sv.ColorPalette.from_hex([
    "#ffff00", "#ff9b00", "#ff8080", "#ff66b2", "#ff66ff", "#b266ff",
    "#9999ff", "#3399ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
])

if os.path.exists(best_model_path):
    print(f"Loading best weights for inference from: {best_model_path}")
    model = RFDETRSegNano(pretrain_weights=best_model_path, resolution=480)
    
    # --- CALCULATE CONVERSION FACTOR FROM REFERENCE IMAGE ---
    print(f"\nAnalyzing bare ground reference image: {ref_img_path}")
    ref_image = cv2.imread(str(ref_img_path))
    if ref_image is None:
        raise ValueError("Could not read the reference image. Please check the path.")

        # Apply Crop to Reference Image for Model Prediction
    if apply_crop and crop_coords is not None:
        xmin, ymin, xmax, ymax = crop_coords
        ref_image = ref_image[ymin:ymax, xmin:xmax]


    ref_detections = model.predict(ref_image)
    
    if len(ref_detections.xyxy) == 0:
        raise ValueError("No pole detected in the reference image!")
        
    # Assuming the first detected object is the pole we want
    x_min_ref, y_min_ref, x_max_ref, y_max_ref = ref_detections.xyxy[0]
    bare_pole_px = y_max_ref - y_min_ref
    
    # # Conversion factor: cm per pixel
    # print(f'{total_pole_cm},{bare_pole_px}')
    # conversion_factor = total_pole_cm / bare_pole_px
    # print(f"-> Reference Pole length in pixels: {bare_pole_px:.2f} px")
    # print(f"-> Calculated Conversion Factor: {conversion_factor:.4f} cm/px")
    # print("-" * 20)

    # --- IMPLEMENTING YOUR IDEA ---
    # If the user didn't know the total pole length (entered NA), 
    # calculate it using the 10cm click conversion factor!
    if total_pole_cm is None:
        print("Total pole length unknown. Calculating it using your calibration clicks...")
        conversion_factor = pixel_centimeter_conversion
        total_pole_cm = bare_pole_px * conversion_factor
        print(f"-> Extrapolated Total Pole Length: {total_pole_cm:.2f} cm")
    else:
        # If they DID provide the total pole length, calculate the conversion factor from it
        conversion_factor = total_pole_cm / bare_pole_px
    
    print(f'Total pole cm: {total_pole_cm:.2f}, Bare pole px: {bare_pole_px:.2f}')
    print(f"-> Reference Pole length in pixels: {bare_pole_px:.2f} px")
    print(f"-> Active Conversion Factor: {conversion_factor:.4f} cm/px")
    print("-" * 20)
    
    # --- RUN PREDICTIONS ON SNOWY IMAGES ---
    images = sorted(glob.glob(os.path.join(camera_image_path, '*.[jJ][pP]*[gG]')))
    sample_images = images #[:5] 
    
    #print(f"\nGenerating predictions for {len(sample_images)} test samples...")
    
    # Set the CSV output path using the dynamic csv directory
    csv_file_path = os.path.join(csv_dir, f"{camera_name}_snowdepth.csv")
    # results_data = [] 
    
    # for i, img_path in tqdm.tqdm(enumerate(sample_images), total=len(sample_images)):
    #     base_name = os.path.basename(img_path)
    #     # print(f"\nProcessing {base_name}...") # Commented out so it doesn't mess up tqdm progress bar
    #     image = cv2.imread(img_path)

    #     formatted_datetime = None
    #     try: 
    #         # 1. Try to get the EXIF creation time by looking for the syntax pattern
    #         with Image.open(img_path) as pil_img:
    #             #IPython.embed()
    #             exif_data = pil_img._getexif()
    #             raw_time = None
                
    #             if exif_data:
    #                 # Regex pattern for exactly "YYYY:MM:DD HH:MM:SS"
    #                 date_pattern = re.compile(r"^\d{4}:\d{2}:\d{2} \d{2}:\d{2}:\d{2}$")
                    
    #                 for value in exif_data.values():
    #                     if isinstance(value, str) and date_pattern.match(value):
    #                         raw_time = value
    #                         break # We found it! Stop searching.
                
    #             if raw_time:
    #                 dt_c = datetime.datetime.strptime(raw_time, '%Y:%m:%d %H:%M:%S')
    #                 formatted_datetime = dt_c.strftime("%m/%d/%Y %H:%M")
    #             else:
    #                 raise ValueError("No matching EXIF time syntax found")
                    
    #     except Exception: 
    #         # 2. Fallback: get the OS modification time if EXIF is missing or corrupted
    #         try:
    #             creationTime = os.path.getmtime(img_path)
    #             dt_c = datetime.datetime.fromtimestamp(creationTime)
    #             formatted_datetime = dt_c.strftime("%m/%d/%Y %H:%M")
    #         except Exception:
    #             formatted_datetime = "Unknown" 

         # ---------------------------------------------------------
    # --- CHECKPOINT/RESUME LOGIC: SKIP PROCESSED IMAGES ---
    # ---------------------------------------------------------
    processed_files = set()
    results_data = [] 
    
    if os.path.exists(csv_file_path):
        print(f"\nFound existing CSV checkpoint at: {csv_file_path}")
        try:
            existing_df = pd.read_csv(csv_file_path)
            if not existing_df.empty and 'filename' in existing_df.columns:
                processed_files = set(existing_df['filename'])
                # Load previously processed files so we don't overwrite them
                results_data = existing_df.to_dict('records') 
                print(f"Loaded {len(processed_files)} previously processed images. Skipping these...")
        except Exception as e:
            print(f"Warning: Could not read existing CSV ({e}). Starting from scratch.")

    # Filter out images we have already processed
    sample_images = [img for img in images if os.path.basename(img) not in processed_files]
    
    if len(sample_images) == 0:
        print(f"\nAll {len(images)} images have already been processed! Displaying summary plot...")
    else:
        print(f"\nGenerating predictions for {len(sample_images)} images...")


    active_anchors = {} # This will store the expected X-coordinate of each pole
    MAX_SHIFT_PX = 1000 # The max pixels a pole can move between frames before it's considered a "new" pole
    
    for i, img_path in tqdm.tqdm(enumerate(sample_images), total=len(sample_images)):
        base_name = os.path.basename(img_path)
        image = cv2.imread(img_path)

                # APPLY CROP IN INFERENCE LOOP
        if apply_crop and crop_coords is not None:
            xmin, ymin, xmax, ymax = crop_coords
            image = image[ymin:ymax, xmin:xmax]

        formatted_datetime = None
        try: 
            # 1. Try to get the EXIF creation time by looking for the syntax pattern
            with Image.open(img_path) as pil_img:
                exif_data = pil_img._getexif()
                raw_time = None
                
                if exif_data:
                    # Regex pattern for exactly "YYYY:MM:DD HH:MM:SS"
                    date_pattern = re.compile(r"^\d{4}:\d{2}:\d{2} \d{2}:\d{2}:\d{2}$")
                    
                    for value in exif_data.values():
                        if isinstance(value, str) and date_pattern.match(value):
                            raw_time = value
                            break # We found it! Stop searching.
                
                if raw_time:
                    dt_c = datetime.datetime.strptime(raw_time, '%Y:%m:%d %H:%M:%S')
                    formatted_datetime = dt_c.strftime("%m/%d/%Y %H:%M")
                else:
                    raise ValueError("No matching EXIF time syntax found")
                    
        except Exception: 
            # 2. Fallback: get the OS modification time if EXIF is missing or corrupted
            try:
                creationTime = os.path.getmtime(img_path)
                dt_c = datetime.datetime.fromtimestamp(creationTime)
                formatted_datetime = dt_c.strftime("%m/%d/%Y %H:%M")
            except Exception:
                formatted_datetime = "Unknown" 
        with torch.no_grad():
            #IPython.embed()
            detections = model.predict(image)
            master_mask = detections.mask[0] if detections.mask is not None and len(detections.mask) > 0 else None
            # 1. Process all detected bounding boxes
            current_detections = []
            for xyxy in detections.xyxy:
                x_min, y_min, x_max, y_max = xyxy
                dx = x_max - x_min
                dy = y_max - y_min
                x_center = (x_min + x_max) / 2 # Find the center of the pole
                
                pole_length_px = math.hypot(dx, dy) if dx > 10 else dy
                visible_length_cm = pole_length_px * conversion_factor
                snow_depth_cm = total_pole_cm - visible_length_cm

                # --- NEW SAFE TILT CALCULATION ---
                ## could add the left tilt right tilt math to consoldate 
                tilt_degrees = 0.0 # Default to 0 (perfectly vertical)
                if dx > 10 and master_mask is not None:
                    # Look at mask inside this bbox
                    y_ind, x_ind = np.where(master_mask)
                    in_box = (x_ind >= int(x_min)) & (x_ind <= int(x_max)) & (y_ind >= int(y_min)) & (y_ind <= int(y_max))
                    y_in = y_ind[in_box]
                    x_in = x_ind[in_box]
                    
                    if len(y_in) > 0:
                        x_top_mask = x_in[np.argmin(y_in)]
                        x_bottom_mask = x_in[np.argmax(y_in)]
                        
                        # Calculate difference in X (dy is already y_max - y_min)
                        dx_tilt = x_bottom_mask - x_top_mask 
                        tilt_radians = math.atan2(dx_tilt, dy)
                        tilt_degrees = math.degrees(tilt_radians)
                
                current_detections.append({
                    'x_center': x_center,
                    'xyxy': xyxy,
                    'snowdepth': snow_depth_cm,
                    'pixellength': pole_length_px,
                    'tilt_degrees': tilt_degrees
                })
                
            # ========================================================
            # 2. MATCHING LOGIC (The Rolling Anchor System)
            # ========================================================
            assigned_poles = {}
            
            if not active_anchors and len(current_detections) > 0:
                # First frame with poles: Initialize anchors from left-to-right
                # current_detections.sort(key=lambda d: d['x_center'])
                # for idx, det in enumerate(current_detections):
                #     pole_id = f"Pole{idx + 1}"
                #     active_anchors[pole_id] = det['x_center'] # Save Anchor!
                #     assigned_poles[pole_id] = det

                             # If we have the user's click coordinate, use it to find the Main Pole (Pole 1)
                if 'clicked_x_center' in locals() and clicked_x_center is not None:
                    # 1. Sort detections by how close they are to the user's calibration click
                    current_detections.sort(key=lambda d: abs(d['x_center'] - clicked_x_center))
                    
                    # 2. Pop the closest detection and force it to be "Pole1"
                    main_det = current_detections.pop(0)
                    active_anchors["Pole1"] = main_det['x_center']
                    assigned_poles["Pole1"] = main_det
                    
                    # 3. Sort the remaining detections Left-to-Right
                    current_detections.sort(key=lambda d: d['x_center'])
                    for idx, det in enumerate(current_detections):
                        pole_id = f"Pole{idx + 2}" # Start numbering at 2
                        active_anchors[pole_id] = det['x_center']
                        assigned_poles[pole_id] = det
                        
                    # Put main det back in the list so it gets saved to the CSV properly
                    current_detections.append(main_det)
                    
                else:
                    # Fallback (If you used presaved configs and didn't click)
                    current_detections.sort(key=lambda d: d['x_center'])
                    for idx, det in enumerate(current_detections):
                        pole_id = f"Pole{idx + 1}"
                        active_anchors[pole_id] = det['x_center']
                        assigned_poles[pole_id] = det
            else:
                ## old anchor set-up doesn't account for the fact that sometimes false positives can come into view
                # # Subsequent frames: Match new detections to existing anchors
                # current_detections.sort(key=lambda d: d['x_center'])
                # used_anchors = set()
                
                # for det in current_detections:
                #     best_match_id = None
                #     min_dist = float('inf')
                    
                #     # Check distances to all known anchors
                #     for p_id, anchor_x in active_anchors.items():
                #         if p_id in used_anchors:
                #             continue
                #         dist = abs(det['x_center'] - anchor_x)
                #         if dist < min_dist:
                #             min_dist = dist
                #             best_match_id = p_id
                    
                #     # If the pole didn't jump insanely far, it's a match!
                #     if best_match_id is not None and min_dist < MAX_SHIFT_PX:
                #         assigned_poles[best_match_id] = det
                #         active_anchors[best_match_id] = det['x_center'] # UPDATE ANCHOR TO NEW POSITION!
                #         used_anchors.add(best_match_id)
                #     else:
                #         # If it shifted too far, or a brand new pole appeared, assign a new ID
                #         new_id = f"Pole{len(active_anchors) + 1}"
                #         active_anchors[new_id] = det['x_center']
                #         assigned_poles[new_id] = det
                #         used_anchors.add(new_id)
                                # Subsequent frames: Global Nearest Neighbor matching

                # 1. Calculate distance from every detection to every known anchor
                matches = []
                for det_idx, det in enumerate(current_detections):
                    for p_id, anchor_x in active_anchors.items():
                        dist = abs(det['x_center'] - anchor_x)
                        if dist < MAX_SHIFT_PX:  # (Make sure MAX_SHIFT_PX is set to ~100)
                            matches.append((dist, det_idx, p_id))
                
                # 2. Sort all possible matches by distance (CLOSEST FIRST)
                matches.sort(key=lambda x: x[0])
                
                used_det_idxs = set()
                used_anchors = set()
                
                # 3. Assign the most obvious/closest poles first
                for dist, det_idx, p_id in matches:
                    # If neither the detection nor the anchor have been used yet, assign them!
                    if det_idx not in used_det_idxs and p_id not in used_anchors:
                        assigned_poles[p_id] = current_detections[det_idx]
                        active_anchors[p_id] = current_detections[det_idx]['x_center'] # UPDATE ANCHOR
                        used_det_idxs.add(det_idx)
                        used_anchors.add(p_id)
                
                # 4. Handle unmatched detections (brand new poles or false positives)
                for det_idx, det in enumerate(current_detections):
                    if det_idx not in used_det_idxs:
                        new_id = f"Pole{len(active_anchors) + 1}"
                        active_anchors[new_id] = det['x_center']
                        assigned_poles[new_id] = det
            # ========================================================
            # 3. Create the single row for this image
            # ========================================================
            
            
            row_data = {
                'camera_id': camera_name,
                'season': camera_season, 
                'location': location_information, 
                'image_directory': str(camera_image_path), 
                'pole_length': total_pole_cm_input,
                'filename': base_name,
                'datetime': formatted_datetime,
                'conversion': pixel_centimeter_conversion,
                'notes': other_info,
                'post_processing_notes': "",  
                'flag': 0
            }
            
            # Add the mapped poles into their correct CSV columns
            for p_id, p_data in assigned_poles.items():
                row_data[f'snowdepth_bbox_{p_id}'] = p_data['snowdepth']
                row_data[f'pixellength_bbox_{p_id}'] = p_data['pixellength']
                row_data[f'tilt_degrees_{p_id}'] = p_data['tilt_degrees'] # adding tilt tracker 

                if p_data['snowdepth'] is not None and p_data['snowdepth'] < -20:
                    row_data['flag'] = 1

            results_data.append(row_data)

        # Annotate
        if i % 1 == 0: ## save every 20 for examples (fixed from "if i % 20:" which skips the 0th and multiples of 20)
            h, w = image.shape[:2]
            thickness = sv.calculate_optimal_line_thickness(resolution_wh=(w, h))
            color_annotator = sv.ColorAnnotator(color=color)
            polygon_annotator = sv.PolygonAnnotator(color=color, thickness=thickness)
            
            annotated_image = image.copy()
            annotated_image = color_annotator.annotate(scene=annotated_image, detections=detections)
            annotated_image = polygon_annotator.annotate(scene=annotated_image, detections=detections)

            # #######
            # if detections.mask is not None:
            #     for mask in detections.mask:
            #         y_indices, x_indices = np.where(mask)
            #         if len(y_indices) > 0:
            #             top_idx = np.argmin(y_indices)
            #             bottom_idx = np.argmax(y_indices)          
            #             x_top, y_top = int(x_indices[top_idx]), int(y_indices[top_idx])
            #             x_bottom, y_bottom = int(x_indices[bottom_idx]), int(y_indices[bottom_idx])
            #             cv2.line(annotated_image, (x_top, y_top), (x_bottom, y_bottom), (98, 0, 255), thickness + 1)
            # ######
            
                   # Draw lines using the Master Mask and BBox Cookie Cutter
            for p in assigned_poles.values():
                x_min, y_min, x_max, y_max = p['xyxy']
                dx = x_max - x_min
                
                # If perfectly vertical, draw straight down the middle
                if dx <= 10:
                    x_center = int((x_min + x_max) / 2)
                    pt1 = (x_center, int(y_min))
                    pt2 = (x_center, int(y_max))
                else:
                    if master_mask is not None:
                        # COOKIE CUTTER: Look at the Master Mask, but ONLY inside this BBox
                        y_ind, x_ind = np.where(master_mask)
                        in_box = (x_ind >= int(x_min)) & (x_ind <= int(x_max)) & (y_ind >= int(y_min)) & (y_ind <= int(y_max))
                        y_in = y_ind[in_box]
                        x_in = x_ind[in_box]
                        
                        if len(y_in) > 0:
                            x_top_mask = x_in[np.argmin(y_in)]
                            x_bottom_mask = x_in[np.argmax(y_in)]
                            
                            # Tilt Check \ vs /
                            if x_top_mask < x_bottom_mask:
                                pt1 = (int(x_min), int(y_min)) # Top-Left
                                pt2 = (int(x_max), int(y_max)) # Bottom-Right
                            else:
                                pt1 = (int(x_max), int(y_min)) # Top-Right
                                pt2 = (int(x_min), int(y_max)) # Bottom-Left
                        else:
                            pt1 = (int(x_min), int(y_min))
                            pt2 = (int(x_max), int(y_max))
                    else:
                        pt1 = (int(x_min), int(y_min))
                        pt2 = (int(x_max), int(y_max))

                cv2.line(annotated_image, pt1, pt2, (98, 0, 255), thickness + 1)

            # Convert OpenCV BGR image to PIL RGB Image for thumbnailing/saving
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(annotated_image_rgb)

            # Create thumbnail and save
            pil_img.thumbnail((800, 800))
        
            out_name = f"pred_{i+1}_{base_name}"
            # Save to the new dynamic sample_outputs directory
            save_path = os.path.join(viz_dir, out_name)

            pil_img.save(save_path)

        ######## checkpoint every 10 and clear memory #####
        if (i + 1) % 10 == 0:
            pd.DataFrame(results_data).to_csv(csv_file_path, index=False)
            ### dump any memory caches ##
            gc.collect() 
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del image
        del detections
        if 'annotated_image' in locals():
            del annotated_image

    df = pd.DataFrame(results_data)
    df.to_csv(csv_file_path, index=False)
    
    print("\n" + "-" * 20)
    print(f"Data saved to CSV: {csv_file_path}")
    print(f"Example visualizations saved to: {viz_dir}")
    print("-" * 20)

    # if not df.empty and 'snowdepth' in df.columns:
    #     print("\nGenerating Snow Depth summary plot...")
    #     plt.figure(figsize=(10, 6))
        
    #     # Plot Bounding Box based snow depth
    #     plt.plot(df.index, df['snowdepth'], marker='o', linestyle='-', color='b', label='Snow Depth (BBox)')
        
    #     # Plot Mask based snow depth if available (so you can compare them!)
    #     if 'snowdepth_mask' in df.columns and not df['snowdepth_mask'].isna().all():
    #         plt.plot(df.index, df['snowdepth_mask'], marker='x', linestyle='--', color='r', alpha=0.7, label='Snow Depth (Mask)')
            
    #     plt.title(f"Estimated Snow Depth over Time/Images - {camera_name}")
    #     max_depth = df['snowdepth'].max()
    #     if pd.notna(max_depth): # Only set ylim if we actually have valid numbers
    #         plt.ylim(0, max_depth + 10) # Added +10cm for a little visual padding at the top

    #     plt.xlabel("Image Index")
    #     plt.ylabel("Snow Depth (cm)")
    #     plt.grid(True, linestyle='--', alpha=0.7)
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()  # This will pause and display the pop-up plot to the user

    if not df.empty and 'snowdepth_bbox_Pole1' in df.columns: ## defaults to first pole (makes sure there is at least one)
        print("\nGenerating Interactive Snow Depth summary plot...")
        
        # 1. Figure out if we should use datetime or index for the X-axis
        # Convert the datetime strings into actual datetime objects for plotting
        parsed_dt = pd.to_datetime(df['datetime'], errors='coerce')
        
        # If we have at least one valid datetime, use datetimes for the X-axis
        use_datetime = not parsed_dt.isna().all()
        x_data = parsed_dt if use_datetime else df.index
        x_label = "Date / Time" if use_datetime else "Image Index"

        # 2. Setup the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # # Plot Bounding Box based snow depth (note the comma after line_bbox)
        # line_bbox, = ax.plot(x_data, df['snowdepth'], marker='o', linestyle='-', color='b', label='Snow Depth (BBox)')
        
        # # Plot Mask based snow depth if available
        # line_mask = None
        # if 'snowdepth_mask' in df.columns and not df['snowdepth_mask'].isna().all():
        #     line_mask, = ax.plot(x_data, df['snowdepth_mask'], marker='x', linestyle='--', color='r', alpha=0.7, label='Snow Depth (Mask)')
                # Plot Bounding Box based snow depth
        #line_bbox, = ax.plot(x_data, df['snowdepth_bbox'], marker='o', linestyle='-', color='b', label='Depth (BBox Overall)')
        
        # line_mask1 = None
        # line_mask2 = None
        
        # # Plot Mask based snow depth for Pole 1 (Left Pole)
        # if 'snowdepth_mask_Pole1' in df.columns and not df['snowdepth_mask_Pole1'].isna().all():
        #     line_mask1, = ax.plot(x_data, df['snowdepth_mask_Pole1'], marker='x', linestyle='--', color='r', alpha=0.7, label='Depth (Left Pole)')
            
        # # Plot Mask based snow depth for Pole 2 (Right Pole)
        # if 'snowdepth_mask_Pole2' in df.columns and not df['snowdepth_mask_Pole2'].isna().all():
        #     line_mask2, = ax.plot(x_data, df['snowdepth_mask_Pole2'], marker='+', linestyle=':', color='g', alpha=0.7, label='Depth (Right Pole)')
        
        line_pole1 = None
        line_pole2 = None
        
        if 'snowdepth_bbox_Pole1' in df.columns and not df['snowdepth_bbox_Pole1'].isna().all():
            line_pole1, = ax.plot(x_data, df['snowdepth_bbox_Pole1'], marker='o', linestyle='-', color='b', label='Depth (First Pole or Left Pole)')
            
        if 'snowdepth_bbox_Pole2' in df.columns and not df['snowdepth_bbox_Pole2'].isna().all():
            line_pole2, = ax.plot(x_data, df['snowdepth_bbox_Pole2'], marker='x', linestyle='--', color='r', alpha=0.7, label='Depth (Right Pole)')
            

            # Updated Title to include instructions
        ax.set_title(f"Estimated Snow Depth over Time/Images - {camera_name}\n(Press 'Enter' to Save & Close)")
            
            # Format X-axis
        ax.set_xlabel(x_label)
        if use_datetime:
            fig.autofmt_xdate() # Rotates the dates so they don't overlap
            
        # Format Y-axis
        #max_depth = df['snowdepth_bbox'].max()
        max_1 = df['snowdepth_bbox_Pole1'].max() if 'snowdepth_bbox_Pole1' in df.columns else 0
        max_2 = df['snowdepth_bbox_Pole2'].max() if 'snowdepth_bbox_Pole2' in df.columns else 0
        max_depth = max(max_1, max_2)
        if pd.notna(max_depth): 
            ax.set_ylim(0, max_depth + 10) 
        ax.set_ylabel("Snow Depth (cm)")
        
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()

        # 3. --- Interactive Hover Tooltip Setup ---
        # Create a hidden annotation box
        annot = ax.annotate("", xy=(0,0), xytext=(15,15), textcoords="offset points",
                            bbox=dict(boxstyle="round4,pad=0.5", fc="white", ec="black", lw=1, alpha=0.9),
                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1"))
        annot.set_visible(False)

        def update_annot(ind, line_obj, method_name):
            # ind["ind"] is a list of the points currently hovered. We grab the first one.
            idx = ind["ind"][0] 
            x_vals, y_vals = line_obj.get_data()
            annot.xy = (x_vals[idx], y_vals[idx])
            
            # Fetch corresponding row data from the DataFrame
            df_idx = df.index[idx]
            row = df.iloc[idx]
            
            filename = row.get('filename', 'Unknown')
            dt_str = row.get('datetime', None)
            
            # Check if datetime is completely missing or 'Unknown'
            if pd.isna(dt_str) or dt_str == "Unknown":
                dt_str = "None"
                
            depth = y_vals[idx]
            
            # Format the text inside the tooltip
            text = f"Index: {df_idx}\nFile: {filename}\nDate: {dt_str}\n{method_name}: {depth:.2f} cm"
            annot.set_text(text)

        # def hover(event):
        #     vis = annot.get_visible()
        #     if event.inaxes == ax:
        #         # Check if mouse is over the BBox line
        #         cont_bbox, ind_bbox = line_bbox.contains(event)
        #         if cont_bbox:
        #             update_annot(ind_bbox, line_bbox, "Depth (BBox)")
        #             annot.set_visible(True)
        #             fig.canvas.draw_idle()
        #             return
                
        #         # # Check if mouse is over the Mask line
        #         # if line_mask is not None:
        #         #     cont_mask, ind_mask = line_mask.contains(event)
        #         #     if cont_mask:
        #         #         update_annot(ind_mask, line_mask, "Depth (Mask)")
        #         #         annot.set_visible(True)
        #         #         fig.canvas.draw_idle()
        #         #         return

        #                      # Check Pole 1
        #         if line_mask1 is not None:
        #             cont_mask1, ind_mask1 = line_mask1.contains(event)
        #             if cont_mask1:
        #                 update_annot(ind_mask1, line_mask1, "Depth (Left Pole)")
        #                 annot.set_visible(True)
        #                 fig.canvas.draw_idle()
        #                 return
                        
        #         # Check Pole 2
        #         if line_mask2 is not None:
        #             cont_mask2, ind_mask2 = line_mask2.contains(event)
        #             if cont_mask2:
        #                 update_annot(ind_mask2, line_mask2, "Depth (Right Pole)")
        #                 annot.set_visible(True)
        #                 fig.canvas.draw_idle()
        #                 return
                
        #         # Hide if not hovering over any line
        #         if vis:
        #             annot.set_visible(False)
        #             fig.canvas.draw_idle()

        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                if line_pole1 is not None:
                    cont1, ind1 = line_pole1.contains(event)
                    if cont1:
                        update_annot(ind1, line_pole1, "Depth (First Pole or Left Pole)")
                        annot.set_visible(True)
                        fig.canvas.draw_idle()
                        return
                        
                if line_pole2 is not None:
                    cont2, ind2 = line_pole2.contains(event)
                    if cont2:
                        update_annot(ind2, line_pole2, "Depth (Right Pole)")
                        annot.set_visible(True)
                        fig.canvas.draw_idle()
                        return
                
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

        # 4. --- Keyboard event for Enter Key ---
        def on_key_press(event):
            if event.key == 'enter':
                # Save the figure to the camera output directory
                save_path = os.path.join(camera_out_dir, f"{camera_name}_summary_plot.png")
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"\n[+] Summary plot saved to: {save_path}")
                plt.close(fig) ### this should shutdown the plot

        # Connect the functions to the figure
        fig.canvas.mpl_connect("motion_notify_event", hover)
        fig.canvas.mpl_connect("key_press_event", on_key_press)
        # ---------------------------------------

        plt.tight_layout()
        plt.show()

# Final completion message (This runs after the plot window is closed)
    print(f"\nProcessing for camera {camera_name} complete!\n")
# else:
    #rint(f"Error: Model not found at {best_model_path}")




    #### TILT math ###

    # # 1. Process all detected bounding boxes
    #         current_detections = []
    #         for xyxy in detections.xyxy:
    #             x_min, y_min, x_max, y_max = xyxy
    #             dx = x_max - x_min
    #             dy = y_max - y_min
    #             x_center = (x_min + x_max) / 2
                
    #             pole_length_px = math.hypot(dx, dy) if dx > 10 else dy
    #             visible_length_cm = pole_length_px * conversion_factor
    #             snow_depth_cm = total_pole_cm - visible_length_cm

    #             # --- CONSOLIDATED TILT & DRAWING MATH ---
    #             tilt_degrees = 0.0 
    #             # Default draw points (straight down the middle)
    #             pt1 = (int(x_center), int(y_min))
    #             pt2 = (int(x_center), int(y_max))
                
    #             if dx > 10 and master_mask is not None:
    #                 y_ind, x_ind = np.where(master_mask)
    #                 in_box = (x_ind >= int(x_min)) & (x_ind <= int(x_max)) & (y_ind >= int(y_min)) & (y_ind <= int(y_max))
    #                 y_in = y_ind[in_box]
    #                 x_in = x_ind[in_box]
                    
    #                 if len(y_in) > 0:
    #                     x_top_mask = x_in[np.argmin(y_in)]
    #                     x_bottom_mask = x_in[np.argmax(y_in)]
                        
    #                     # 1. Calculate Tilt
    #                     dx_tilt = x_bottom_mask - x_top_mask 
    #                     tilt_radians = math.atan2(dx_tilt, dy)
    #                     tilt_degrees = math.degrees(tilt_radians)

    #                     # 2. Determine Drawing Points (Lean Direction)
    #                     if x_top_mask < x_bottom_mask:
    #                         pt1 = (int(x_min), int(y_min)) # Top-Left
    #                         pt2 = (int(x_max), int(y_max)) # Bottom-Right
    #                     else:
    #                         pt1 = (int(x_max), int(y_min)) # Top-Right
    #                         pt2 = (int(x_min), int(y_max)) # Bottom-Left
                
    #             current_detections.append({
    #                 'x_center': x_center,
    #                 'xyxy': xyxy,
    #                 'snowdepth': snow_depth_cm,
    #                 'pixellength': pole_length_px,
    #                 'tilt_degrees': tilt_degrees,
    #                 'draw_pt1': pt1,    # <--- Save drawing points
    #                 'draw_pt2': pt2     # <--- Save drawing points
    #             })