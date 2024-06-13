# Steps to Run Code

## Clone the Repository

To clone the repository, run the following command in your terminal:

```bash
git clone https://github.com/asavaribhelawe/Object-detection-and-counting
```
## Go to the Cloned Folder
Navigate into the cloned repository folder:
```bash
cd Object-detection-and-counting
```
## Install Requirements
Install the required Python packages by running:
```bash
pip install -r requirements.txt
```
## Download the Pre-trained YOLOv9 Model Weights
Download the YOLOv9 weights from the following link and place them in your project directory:
[Download YOLOv9 weights](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt)

## Downloading the DeepSORT Files
Download the DeepSORT files by running the following command:
```bash
gdown "https://drive.google.com/uc?id=11ZSZcG-bcbueXZC3rN08CM0qqX3eiHxf&confirm=t"
```
After downloading the DeepSORT Zip file from the drive, unzip it by running the script.py file in yolov9 folder.

## Download Sample Videos
Download the sample videos from the following links:
```bash
gdown "https://drive.google.com/uc?id=115RBSjNQ_1zjvKFRsQK2zE8v8BIRrpdy&confirm=t"
gdown "https://drive.google.com/uc?id=1rjBn8Fl1E_9d0EMVtL24S9aNQOJAveR5&confirm=t"
```
## Running the Code
For Detection Only
To perform detection only, use the following command:

```bash
python detect_dual.py --weights 'yolov9-c.pt' --source 'your video.mp4' --device 0
```
## For Detection and Tracking
For detection and tracking, run:
```bash
python detect_dual_tracking.py --weights 'yolov9-c.pt' --source 'your video.mp4' --device 0
```
## For Specific Class (Person)
To detect and track specific classes (e.g., person), use:
```bash
python detect_dual_tracking.py --weights 'yolov9-c.pt' --source 'your video.mp4' --device 0 --classes 0
```
## For Detection and Tracking with Trails
For detection and tracking with trails, run:
```bash
python detect_dual_tracking.py --weights 'yolov9-c.pt' --source 'your video.mp4' --device 0 --draw-trails
```
## Object Counting
The counting initiates when the trail of the object detected intersects with the line, so make sure you run the command accordingly.

Output files will be created in the working-dir/runs/detect/obj-tracking directory with the original filename.
