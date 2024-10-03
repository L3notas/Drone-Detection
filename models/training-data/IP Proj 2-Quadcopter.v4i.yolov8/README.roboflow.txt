
IP Proj 2-Quadcopter - v4 2024-07-16 3:43pm
==============================

This dataset was exported via roboflow.com on July 17, 2024 at 9:39 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 17081 images.
Quadcopter-fixedwing-laptop-pen are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 2 versions of each source image:
* Random rotation of between -9 and +9 degrees
* Random shear of between -0° to +0° horizontally and -12° to +12° vertically
* Random brigthness adjustment of between -14 and +14 percent
* Random exposure adjustment of between -11 and +11 percent
* Salt and pepper noise was applied to 0.49 percent of pixels


