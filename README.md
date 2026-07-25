<div align="center"> <h1>Snowpole Snow Depth Estimation Tool </h1> 

<p><i>Automated snow depth extraction from trail cameras using Roboflow's RFDETR</i></p> 
</div> <hr>

This script processes a time series of trail camera images to automatically detect snow poles and estimate the depth of the snow. Instead of manually clicking through hundreds of images, this tool uses a fine-tuned RF-DETR (Robust Feature Detection Transformer) model to identify the visible portion of the snow pole, calculates the snow depth based on a physical conversion factor, and exports the data straight to a csv. 

<p>Some Key Features </p> 
- Automated Calibration: Click the top and bottom of a known object in a snow-free image to automatically calculate the pixel-to-centimeter scale.
- Ready-to-Analyze Data: Outputs a clean Pandas dataframe (.csv) with snow depth in centimeters.
- Visual QA/QC: Automatically saves bounding box visualizations for a subset of images so you can verify the model is working correctly.



<p>Prerequisites & Installation </p> 

1. Environment Setup
You must run this inside a Conda or Python environment that has the required packages installed.

bash

Copy code
conda activate rfdetr_snow

2. Required Libraries
Make sure your environment has the following dependencies installed:

<ul> <li><code>numpy</code> and <code>pandas</code></li> <li><code>opencv-python</code> (cv2)</li> <li><code>Pillow</code> (PIL)</li> <li><code>matplotlib</code></li> <li><code>supervision</code></li> <li><code>tqdm</code></li> <li><code>rfdetr</code> <i>(Ensure this module is accessible in your Python path)</i></li> </ul>
3. Model Weights
The script requires a trained RF-DETR model checkpoint. By default, the script looks for this specific file path:
<kbd>/Users/cmbreen/code/snowpoles_rfdetr/checkpoint_best_total.pth</kbd><br>
(Note: If you clone this to a different computer, you will need to update the best_model_path variable in the script to point to your downloaded .pth file).

🚀 How to Run the Code
Open your terminal, navigate to the directory containing the script, and run:

bash

Copy code
python deploy_model.py

The Configuration Wizard
Upon running, the script will ask if you want to use presaved configurations.
If you type <kbd>N</kbd> (First Time / Interactive Mode), you will be prompted to enter:

Camera ID: Name of the site (e.g., TLS-A1N).
Water Year: The season the photos were taken (e.g., 2016-2017).
Image Directory Path: The full path to the folder containing your raw images.
Total Pole Length (cm): The actual length of the full bare pole in centimeters (e.g., 304.8).
Pixel-to-Centimeter Conversion: Type <kbd>NA</kbd> to have the script calculate this automatically.
Reference Image: Indicate if the 1st image in the folder is bare ground (snow-free). If not, provide the image number (e.g., 9).
Output Directories: Paths where your final .csv and visualization images will be saved.
💡 Manual Calibration Tip:
If you set the conversion factor to NA, a window will pop up showing the reference image. Click exactly TWO points (the top and bottom of your bare pole). The script will calculate the physical scale automatically!

📂 What to Expect (Outputs)
When the script finishes processing your images, check your output folders for two things:

📄 Snow Depth Data ([Camera_ID]_snowdepth.csv)
Columns include: camera_id, filename, snowdepth, pixellength, and the conversion factor.
🖼️ Visualizations (pred_*.jpg)
The script saves an annotated thumbnail for every 20th image. You can flip through these to make sure the AI accurately placed a bounding box around the pole!
🆘 Troubleshooting
<details> <summary><b>🔴 WARNING: No images found in [path]!</b></summary> <blockquote>Ensure your file path is exactly correct, contains no typos, and is <b>not</b> wrapped in extra quotes or leading spaces when pasted into the terminal.</blockquote> </details> <details> <summary><b>🔴 TypeError: ufunc 'divide' not supported...</b></summary> <blockquote>Ensure you are entering a numeric value (like <code>304.8</code>) when asked for the pole length. If you accidentally type text (like "2016-2017"), the math calculations will fail!</blockquote> </details> <details> <summary><b>🔴 Error: Model not found at [path]</b></summary> <blockquote>Check that your <code>checkpoint_best_total.pth</code> file actually exists at the exact path defined in the <code>best_model_path</code> variable inside the script.</blockquote> </details>







