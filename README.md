# Real-Time-Human-Detection-and-Counting

When trying to analyse how people are distributed in a crowd, specifically during an event being held it can be very difficult to keep track of where each individual is. With advancements in Deep Learning becoming mainstream it is now possible to gather inferences about the environment that can help us keep track of even the slightest movement. Here people tracking video streams are undertaken with the assistance of Convolutional Neural Networks, leveraging on the YOLOv3 model pre-trained on COCO dataset. By tracking people's bounding boxes and counting how often they show up and disappear, an estimation of the density of the crowd can be made. This can be used to determine the number of people being present in an area, and in turn decide if the area is safely allowing for the maximum number of people present (in accordance with safety regulations as a form of crowd-control). The OpenCV python library was used in conjunction with a Yolov3 object detection algorithm to perform this task.

Real Time Human Detection and Counting with OpenCV and YOLO

## Pre-requisites
- OpenCV-python
- Numpy
- imutils
- Argparse
- dlib
- Sklearn

## Usage
```
python human_counter.py \
    -y [Path to YOLO directory] \
    -i [Path to input video]\
 	--tl 20 \
 	--hide_output 0
```
## Demo videos:
- [Crowd Density and event count over a period of 2 hours ](https://youtu.be/4Cb0diwBChU)
- [Crowd Distribution Free Movement around a museum ](https://youtu.be/XqnE54ItJmk)
- [Crowd Distribution during an Robotic Development Competition](https://youtu.be/cqMFB8ghtvQ)

## Output
  Output will be displayed in a new window created by OpenCV, but can also be viewed in the videos above or inside the Output directory. 

## Limitations:
Current limitations of the detection algorithm can be found inside the `custom_yolo_model` directory.



