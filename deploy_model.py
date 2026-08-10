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
from matplotlib.widgets import Button  # Added for the Reset button

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
    images = glob.glob(os.path.join(camera_image_path, '*.[jJ][pP]*[gG]'))
    sample_images = images #[:5] 
    
    print(f"\nGenerating predictions for {len(sample_images)} test samples...")
    
    # Set the CSV output path using the dynamic csv directory
    csv_file_path = os.path.join(csv_dir, f"{camera_name}_snowdepth.csv")
    results_data = [] 
    
    for i, img_path in tqdm.tqdm(enumerate(sample_images)):
        base_name = os.path.basename(img_path)
        # print(f"\nProcessing {base_name}...") # Commented out so it doesn't mess up tqdm progress bar
        image = cv2.imread(img_path)
        
        detections = model.predict(image)
        # 1. Safety check: create a list of 'None' if masks are missing
        masks = detections.mask if detections.mask is not None else [None] * len(detections.xyxy)
        # 2. Correctly unpack the index (j) and the zipped items (xyxy, mask)
        for j, (xyxy, mask) in enumerate(zip(detections.xyxy, masks)):
            x_min, y_min, x_max, y_max = xyxy
            # Calculate vertical and horizontal differences
            dy = y_max - y_min
            dx = x_max - x_min
            # If the pole is tilted (bounding box is wider than 10 pixels)
            # use the hypotenuse 
            if dx > 10:
                pole_length_px = math.hypot(dx, dy) 
            else:
                pole_length_px = dy

            visible_length_cm = pole_length_px * conversion_factor
            snow_depth_cm = total_pole_cm - visible_length_cm

            ####################
            #   # Get the row (y) and column (x) coordinates of all pixels that make up the pole
            # y_indices, x_indices = np.where(mask)
            # if len(y_indices) == 0:
            #     continue
            # top_idx = np.argmin(y_indices)
            # x_top, y_top = x_indices[top_idx], y_indices[top_idx]
            # bottom_idx = np.argmax(y_indices)
            # x_bottom, y_bottom = x_indices[bottom_idx], y_indices[bottom_idx]
            # dx = x_bottom - x_top
            # dy = y_bottom - y_top
            # pole_length_px = math.hypot(dx, dy)
            # visible_length_cm = pole_length_px * conversion_factor
            # snow_depth_cm_mask = total_pole_cm - visible_length_cm
            snow_depth_cm_mask = None
            pole_length_px_mask = None
            
            if mask is not None:
                y_indices, x_indices = np.where(mask)
                if len(y_indices) > 0:  # Only proceed if the mask isn't empty
                    top_idx = np.argmin(y_indices)
                    x_top, y_top = x_indices[top_idx], y_indices[top_idx]
                    
                    bottom_idx = np.argmax(y_indices)
                    x_bottom, y_bottom = x_indices[bottom_idx], y_indices[bottom_idx]
                    
                    dx_mask = x_bottom - x_top
                    dy_mask = y_bottom - y_top
                    
                    pole_length_px_mask = math.hypot(dx_mask, dy_mask)
                    visible_length_cm_mask = pole_length_px_mask * conversion_factor
                    snow_depth_cm_mask = total_pole_cm - visible_length_cm_mask
            ####################


            # results_data.append({
            #     'camera_id': camera_name,
            #     'season': , 
            #     'location':, 
            #     'image_directory':, 
            #     'pole_length'
            #     'filename': base_name,
            #     'snowdepth': snow_depth_cm,
            #     'pixellength': pole_length_px,
            #     'conversion': conversion_factor
            #     'notes':
            # })
            results_data.append({
                'camera_id': camera_name,
                'season': camera_season, 
                'location': location_information, 
                'image_directory': str(camera_image_path), 
                'pole_length': total_pole_cm_input,
                'filename': base_name,
                'snowdepth': snow_depth_cm,
                'snowdepth_mask': snow_depth_cm_mask,
                'pixellength': pole_length_px,
                'pixellength_mask': pole_length_px_mask,
                'conversion': pixel_centimeter_conversion,
                'notes': other_info
            })
        
        # Annotate
        if i % 20 == 0: ## save every 20 for examples (fixed from "if i % 20:" which skips the 0th and multiples of 20)
            h, w = image.shape[:2]
            thickness = sv.calculate_optimal_line_thickness(resolution_wh=(w, h))
            color_annotator = sv.ColorAnnotator(color=color)
            polygon_annotator = sv.PolygonAnnotator(color=color, thickness=thickness)
            
            annotated_image = image.copy()
            annotated_image = color_annotator.annotate(scene=annotated_image, detections=detections)
            annotated_image = polygon_annotator.annotate(scene=annotated_image, detections=detections)

            #######
            if detections.mask is not None:
                for mask in detections.mask:
                    y_indices, x_indices = np.where(mask)
                    if len(y_indices) > 0:
                        top_idx = np.argmin(y_indices)
                        bottom_idx = np.argmax(y_indices)          
                        x_top, y_top = int(x_indices[top_idx]), int(y_indices[top_idx])
                        x_bottom, y_bottom = int(x_indices[bottom_idx]), int(y_indices[bottom_idx])
                        cv2.line(annotated_image, (x_top, y_top), (x_bottom, y_bottom), (98, 0, 255), thickness + 1)
                        # # Draw little yellow dots at the exact top and bottom points for visual proof
                        # cv2.circle(annotated_image, (x_top, y_top), thickness + 2, (0, 255, 255), -1) 
                        # cv2.circle(annotated_image, (x_bottom, y_bottom), thickness + 2, (0, 255, 255), -1)
            ######
            
            # Convert OpenCV BGR image to PIL RGB Image for thumbnailing/saving
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(annotated_image_rgb)

            # Create thumbnail and save
            pil_img.thumbnail((800, 800))
        
            out_name = f"pred_{i+1}_{base_name}"
            # Save to the new dynamic sample_outputs directory
            save_path = os.path.join(viz_dir, out_name)

            pil_img.save(save_path)

    df = pd.DataFrame(results_data)
    df.to_csv(csv_file_path, index=False)
    
    print("\n" + "-" * 20)
    print(f"Data saved to CSV: {csv_file_path}")
    print(f"Example visualizations saved to: {viz_dir}")
    print("-" * 20)
else:
    print(f"Error: Model not found at {best_model_path}")