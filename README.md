<div align="center"> <h1>Snowpole Snow Depth Estimation Tool </h1> 

<p><i>Automated snow depth extraction from trail cameras using Roboflow's RFDETR</i></p> 
</div> <hr>

This script processes a time series of trail camera images to automatically detect snow poles and estimate the depth of the snow. Instead of manually clicking through hundreds of images, this tool uses a fine-tuned RF-DETR (Robust Feature Detection Transformer) model to identify the visible portion of the snow pole, calculates the snow depth based on a physical conversion factor, and exports the data straight to a csv. 


<img src="https://github.com/catherine-m-breen/snowpoles_rfdetr/blob/main/pred_5_TLS-A1N_WSCT1734.JPG"> 


##### Note: It helps if you know: height of pole (in cm), location, and camera ID! 

## Some Key Features
<p>- Automated calibration: Click the top and bottom of a snow-free image to automatically calculate the pixel-to-centimeter scale.</p> 
<p>- Snow Depth csv: Outputs a clean Pandas dataframe (.csv) with snow depth in centimeters.</p> 
<p>- Example photos: Automatically saves bounding box visualizations for a subset of images so you can verify the model is working correctly.</p> 


## Prerequisites & Installation 

### Step 1: Initial Setup
1. **Download VSCode:** If you are new to Windows and Python, please see our tutorial. If you have done this before, skip to step 2. 
2. **Create a Workspace:** In VSCode, create a new folder that will be your code folder. Call it `snowpoles_code`.
3. **Download Files:** Download the following files from this repository and place them into your `snowpoles_code` folder: 
    * `requirements.txt` (this tells Python which packages to download)
    * `deploy_model.py` (the actual code)
4. **Open Terminal:** Now we will load the packages from the "terminal" of VSCode. Look at the top menu, click **Terminal -> New Terminal**. A window will open at the bottom of your screen. 
    * *Note for Windows users:* Make sure your terminal is set to "Command Prompt", not "PowerShell".
5. **Download Conda:** If you don't have Conda installed, please download and install [Miniconda](https://docs.anaconda.com/free/miniconda/index.html) for your specific operating system (Mac or Windows).

### Step 2: Creating the Environment
An "environment" is like an isolated sandbox where we put all the specific tools this code needs, without messing up the rest of your computer.

In your VSCode terminal, run this command to create a new Conda environment named `rfdetr_snow` using Python 3.9:
```
conda create -n rfdetr_snow python=3.9 -y
```

### Step 3: Activating and Installing Packages
Once Conda finishes creating the environment, you need to "turn it on" (activate it). Run:

```
conda activate rfdetr_snow
```

(You should now see (rfdetr_snow) at the start of your terminal line).

Now, tell Python to install all the tools listed in the requirements file by running:

```
pip install -r requirements.txt
```

What is being installed?
The requirements.txt file automatically handles installing these dependencies for you:

- numpy and pandas
- opencv-python (cv2)
- Pillow (PIL)
- matplotlib
- supervision
- tqdm
- rfdetr (Ensure this custom module is accessible in your Python folder if not included in pip)


### Step 4: Running the model 
Make sure your environment is activated (conda activate rfdetr_snow), then run the script:
```
python deploy_model.py
```

*The Configuration Set-Up*

Upon running, the script will ask if you want to use presaved configurations (in terminal!)
If you type <kbd>N</kbd> (First Time / Interactive Mode), you will be prompted to enter:

- Camera ID: Name of the site (e.g., TLS-A1N).
- Water Year: The season the photos were taken (e.g., 2016-2017).
- Image Directory Path: The full path to the folder containing your raw images.
- Total Pole Length (cm): The actual length of the full bare pole in centimeters (e.g., 304.8).
- Pixel-to-Centimeter Conversion: Type <kbd>NA</kbd> to have the script calculate this automatically.
- Reference Image: Indicate if the 1st image in the folder is bare ground (snow-free). If not, provide the image number (e.g., 9).
- Output Directories: Paths where your final .csv and visualization images will be saved.
- Manual Calibration Tip:

    If you set the conversion factor to NA, a window will pop up showing the reference image. Click exactly TWO points (the top and bottom of your bare pole). The script will calculate the physical scale automatically!

## What to Expect (Outputs)
When the script finishes processing your images, check your output folders for two things:

- Snow Depth Data ([Camera_ID]_snowdepth.csv)
- Columns include: camera_id, filename, snowdepth, pixellength, and the conversion factor.
-  Visualizations (pred_*.jpg)

The script saves an annotated thumbnail for every 20th image. You can flip through these to make sure the AI accurately placed a bounding box around the pole!

##  Troubleshooting

Possible Errors: 

- "conda: command not found." 
You need to install Miniconda/Anaconda, or you need to restart VSCode after installing it.
- WARNING: No images found in [path]!</b></summary> <blockquote>
Ensure your file path is exactly correct, contains no typos, and is <b>not</b> wrapped in extra quotes or leading spaces when pasted into the terminal.</blockquote> </details> <details> <summary><b>
- TypeError: ufunc 'divide' not supported...</b></summary> <blockquote>Ensure you are entering a numeric value (like <code>304.8</code>) when asked for the pole length. If you accidentally type text (like "2016-2017"), the math calculations will fail!</blockquote> </details> <details> <summary><b> 
- Error: Model not found at [path]</b></summary> <blockquote>Check that your <code>checkpoint_best_total.pth</code> file actually exists at the exact path defined in the <code>best_model_path</code> variable inside the script.</blockquote> </details>
- NameError: name 'ref_img_path' is not defined </b></summary> <blockquote> This error happens when the script cannot find your images to set the reference image, usually because the folder path you entered has a hidden mistake. Double-check that there are no leading spaces when you paste the image directory path into the terminal.</blockquote> </details> 








