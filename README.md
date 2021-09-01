# OrbitalObjectDetector

![Telescope image showing object detection prediction](https://user-images.githubusercontent.com/65425526/131582315-55b94b91-521b-4f3f-a08f-3b0e5105d14a.png)

## Object detection for Liverpool Telescope images of Earth-orbiting objects using Faster-RCNN from Detectron2.

### Requirements

To use this program you will need to install Detectron2 on your system. Windows users can follow this tutorial:
https://dgmaxime.medium.com/how-to-easily-install-detectron2-on-windows-10-39186139101c

### Usage

The program is provided with an example dataset and model+configuration. Run 'detect.py' to see example outputs predicted by Faster-RCNN.
The results of detect.py are written to the 'annotated results' folder in the format of a copy of each input image with the prediction overlayed,
and a .txt file containing the pixel coordinates of the corners of the bounding box for the object.

The algorithm can be trained on a custom dataset by putting your data and JSON annotation files in the 'dataset' folder and running 'train.py'.

_This program is in development as part of a processing chain for light level measurement of Earth-orbiting objects._
_The program to prepare data from the FITS and XLSX files is available at: https://github.com/KaneLindsay/COCOFormatter _
