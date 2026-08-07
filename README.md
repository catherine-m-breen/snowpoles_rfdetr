<div align="center"> <h1>Snowpole Snow Depth Estimation Tool </h1> 

<p><i>Automated snow depth extraction from trail cameras using Roboflow's RFDETR</i></p> 
</div> <hr>

This script processes a time series of timelapse camera images to automatically detect snow poles and estimate the depth of the snow. Instead of clicking through images, this tool uses a fine-tuned version of the RF-DETR (Robust Feature Detection Transformer) model to identify the visible portion of the snow pole, calculate the snow depth based on a pixel-to-centimeter converesion factor, and then export the data to a csv. 


<img src="https://github.com/catherine-m-breen/snowpoles_rfdetr/blob/main/sample_images/pred_5_TLS-A1N_WSCT1734.JPG"> 


##### Note: It helps if you know: height of pole (in cm), location, and camera ID! 

## Some Key Features
<p>- Automated calibration: Click the top and bottom of a snow-free image to automatically calculate the pixel-to-centimeter scale.</p> 
<p>- Snow Depth csv: Outputs a clean Pandas dataframe (.csv) with snow depth in centimeters.</p> 
<p>- Example photos: Automatically saves bounding box visualizations for a subset of images so you can verify the model is working correctly.</p> 


## Prerequisites & Installation 
 

### Step 1: Initial Setup
1. **Download Softwares:** You will need the following softwares to use this tool. Use the documents below for each software to download. If you already have the softwares you can skip to step 1.2. 
    - VSCode: https://code.visualstudio.com/download?_exp_download=d53503e735 
    - Git: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git 
    - Miniconda: 
        - Windows: https://www.anaconda.com/docs/getting-started/miniconda/install/windows-gui-install
        - Mac: https://www.anaconda.com/docs/getting-started/miniconda/install/mac-gui-install 
2. **Create a Workspace:** If you have not already, download VSCode from the link in Step 1. Now, in VSCode, create a new folder that will be your code folder. Call it `snowpoles_code`.
3. **Open Terminal:** Now we will load the packages from the "terminal" of VSCode. Look at the top menu, click **Terminal -> New Terminal**. A window will open at the bottom of your screen. 
    - For Mac users: Command + J will also open a terminal window at the bottom of the screen. 
    - For Window users: Ctrl + J (or Ctrl + ` ) will also open the terminal. Important: Your terminal might default to "PowerShell", but we need it to be "Command Prompt". To change this, look at the top right of the terminal window, click the downward arrow next to the + icon, and select Command Prompt.

    * *Note for Windows users:* Make sure your terminal is set to "Command Prompt", not "PowerShell". You will know you are in Command Prompt / Terminal by the word "Terminal" at the top left corner of your terminal window. STOP if it still says Power Shell, and make sure to switch to Command Prompt/ Terminal by using the downward arrow next to the '+' icon on the far right side of the window and switching to Command Prompt.

4. **Download Files:** Download the code folder from this repository and place them into your `snowpoles_code` folder: 

    You can download the files two ways. Either by 1. downloading from the green button in the top corner of this github repository or doing a 'git clone.' We recommend a git clone as this is more coding friendly and will allow you to pull any updates to the model that come through. If you do not have Git installed, go back to Step 1, and follow the instructions to download Git. 
    
    After ensuring you have git installed, run the followng commands:

    ```
    git clone https://github.com/catherine-m-breen/snowpoles_rfdetr.git
    cd snowpoles_rfdetr
    ```

    On the left-hand side of your screen you will now see several files in your file "Explorer." The main ones we will be using are: 
    * `environment.yml` (this tells Python which packages to download)
    * `deploy_model.py` (the actual code)
    

### Step 2: Creating the Environment
An "environment" is like an isolated sandbox where we put all the specific tools this code needs, without messing up the rest of your computer or other coding environments that might have different package requirements. Ideally for projects, each has their own "coding sandbox" or basically conda environemnt. 

If you don't have Conda installed, please download and install [Miniconda](https://docs.anaconda.com/free/miniconda/index.html) for your specific operating system (Mac or Windows). See Step 1.1.

In your VSCode terminal, run this command to create a new Conda environment named `rfdetr_snow` using Python 3.14:

```
conda env create -f environment.yml
```

Note: this will take several minutes to download (estimate: 3-4 minutes)

What is being installed?
The environment.yml file automatically handles installing these dependencies for you:

- numpy and pandas
- opencv-python (cv2)
- Pillow (PIL)
- matplotlib
- supervision
- tqdm
- rfdetr (Ensure this custom module is accessible in your Python folder if not included in pip)


### Part 3: Activating and Installing Packages
Once Conda finishes creating the environment, you need to "turn it on" (activate it). Run:

```
conda activate rfdetr_snow
```

(You should now see (rfdetr_snow) at the start of your terminal line).


### Step 4: Running the model 
Make sure your environment is activated (conda activate rfdetr_snow), then run the script:
```
python deploy_model.py
```

*The Configuration Set-Up*

Upon running, the script will ask if you want to use presaved configurations (in terminal!) Most of the time you will press <kbd>N</kbd> (First Time / Interactive Mode), because you are running the code on a camera folder for the first time to do processing. Pressing <kbd>Y</kbd> is the advanced mode and would be for use cases where someone wants to fiddle with the "saved_configs" variable at the top of the script so they don't have to enter in the information interactively every time (would require knowing the information in advance). However, again, most users will use <kbd>N</kbd> (First Time / Interactive Mode)! 

If you type <kbd>N</kbd> (First Time / Interactive Mode), you will be prompted to enter:

- Camera ID: Name of the site (e.g., TLS-A1N, Camera1, E9E, etc). This does not need to be in a string, just entering it is fine. Spaces are also fine, although, best practice in coding is to not use spaces. 
- Water Year: The season the photos were taken (e.g., 2016-2017). This also does not need to be in a string, just entering the water or years is fine. The model saves in a column called "Season." If you don't know you can put NA. 
- Image Directory Path: The full path to the folder containing your raw images. This is likely the hardest part as the path name needs to be exactly right for the computer to know where to find it. 
    - For Mac: Navigate to the folder on your computer and on the folder name, right click while holding down the <kbd>Option</kbd> key. You'll see an option about halfway down called "Copy [Camera Folder] as Pathname." The Pathname is now copied to your folder in the correct format. 
    - For Windows: Navigate to the folder on your computer and hold down the 'Shift' key while right-clicking the folder. Select "Copy as path" from the menu. (Alternatively, you can open the folder, click on the address bar at the top of the File Explorer window so the path highlights in blue, and press <kbd>Ctrl+C</kbd> to copy it).
- Total Pole Length (cm): The actual length of the full bare pole in centimeters (e.g., 304.8). The model works best if you know the full length of the pole, but it will also work if you just have a calibration segment (a 10cm section or similar) that you do know the length of. Basically, you need to know the length of some or all of the pole. Put the length of the pole here, or write NA. 
- Pixel-to-Centimeter Conversion: Type <kbd>NA</kbd> to have the script calculate this automatically. This will be done in the next step. Advanced: If you have run the script many times for the same camera folder, you may know the pixel-to-centimeter conversion in which case you could enter it here to use the same one every time. 
- Reference Image: Indicate if the 1st image in the folder is bare ground (snow-free). If not, provide the image number (e.g., 9). Sometimes the first couple images are from set-up. Typically these images should get cleaned out of your timelapse camera folder. However, if they are still in there enter the correct image that should be the reference image. The model will use that image as the reference for all images in the folder. 
- Output Directories: Paths where your final .csv and visualization images will be saved. The default is to press <kbd>enter</kbd> . Doing so creates a default folder called 'outputs' in your workspace folder. 

* Manual Calibration*

If you set the conversion factor to NA during set-up two things will happen: 

1. The terminal will ask you: 'Enter the known length in cm you will click (e.g., total pole length or 10):" This means what you want to use to find your conversion factor between pixels and centimeter. We find that many people have many different pole designs. Most often people know the full length of their pole (lke if it's 2 m tall or similar), or they have added a calibration segment (such as a 10-cm yellow or some other bright colored top). In Breen et al. (2023) they found that the calibrating with the full length of the pole is more accurate than a similar segment, because it's easier to click the top and bottom of the pole than the top and bottom of the smaller calibriation segment. So we recommend that whenever possible to enter the total pole length in centimeters and use that for calibration. However, if you're pole design only has a 10-cm top or similar that will work here too, and enter whatever the length of the smaller calibration section is. 

2. After entering that information, an image will pop up showing the reference image (whatever number you entered as the image that should be treated as the reference image). Click exactly TWO points that correspond to whatever the length in centimeters you entered in the terminal answer above. This is usually the top and bottom of your bare pole. The script does the following calculation 

pixel to centimeter conversion = length in centimeters / length in pixels 

By figuring out the conversion factor between centimeters the model is now ready to make the predictions in pixels and then convert to centimeters. 

## What to Expect (Outputs)
When the script finishes processing your images, check your output folders for two things:

- Snow Depth Data ([Camera_ID]_snowdepth.csv)
- Columns include: camera_id, filename, snowdepth, pixellength, and the conversion factor.
-  Visualizations (pred_*.jpg)

The script saves an annotated thumbnail for every 20th image. You can flip through these to make sure the model accurately placed a bounding box around the pole. There will likely be some buffer around the poles due to labeling framework (we converted keypoints to segmented masks)

##  Troubleshooting

Possible Errors: 

- "conda: command not found." 
You need to install Miniconda/Anaconda, or you need to restart VSCode after installing it.
- WARNING: No images found in [path]!</b></summary> <blockquote>
Ensure your file path is exactly correct, contains no typos, and is <b>not</b> wrapped in extra quotes or leading spaces when pasted into the terminal.</blockquote> </details> <details> <summary><b>
- TypeError: ufunc 'divide' not supported...</b></summary> <blockquote>Ensure you are entering a numeric value (like <code>304.8</code>) when asked for the pole length. If you accidentally type text (like "2016-2017"), the math calculations will fail!</blockquote> </details> <details> <summary><b> 
- Error: Model not found at [path]</b></summary> <blockquote>Check that your <code>checkpoint_best_total.pth</code> file actually exists at the exact path defined in the <code>best_model_path</code> variable inside the script.</blockquote> </details>
- NameError: name 'ref_img_path' is not defined </b></summary> <blockquote> This error happens when the script cannot find your images to set the reference image, usually because the folder path you entered has a hidden mistake. Double-check that there are no leading spaces when you paste the image directory path into the terminal.</blockquote> </details> 








