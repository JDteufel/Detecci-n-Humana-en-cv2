# Real-Time-Human-Detection-and-Counting

Real Time Human Detection and Counting with OpenCV and YOLOV3.
The model used for detection is YOLOV3. All weights can be downloaded here: https://pjreddie.com/darknet/yolo/ or from this repo: yolov3.weights -> Download file (254 MB).

For accurate results, use a video...


## Pre-requisites
- OpenCV-python
- Numpy
- imutils
- Argparse
- Sklearn

## Usage
Use git clone to clone the repo and other pre-requisites.
Install and Add to PATH [OpenCV (with contrib modules)](https://www.pyimagesearch.com/opencv-tutorials-resources-guides/) together with other pre-requisites.

## Demo videos:
- [Crowd Density and event count over a period of 2 hours ](https://youtu.be/4Cb0diwBChU)
- [Crowd Distribution Free Movement around a museum ](https://youtu.be/XqnE54ItJmk)
- [Crowd Distribution during an Robotic Development Competition](https://youtu.be/cqMFB8ghtvQ)

## Output
  Output will be displayed in a new window created by OpenCV, but can also be viewed in the videos above or inside the Output directory. 

## Limitations:
Current limitations of the detection algorithm can be found inside the `custom_yolo_model` directory.



