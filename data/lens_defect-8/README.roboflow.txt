
lens_defect - v8 2025-11-11 4:52pm
==============================

This dataset was exported via roboflow.com on October 11, 2026 at 6:50 PM GMT

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

The dataset includes 551 images.
Collapse are annotated in COCO Segmentation format.

The following pre-processing was applied to each image:
* Resize to 2048x2048 (Fill (with center crop))
* Auto-contrast via adaptive equalization

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise, upside-down
* Random exposure adjustment of between -16 and +16 percent
* Random Gaussian blur of between 0 and 0.5 pixels
* Salt and pepper noise was applied to 1.45 percent of pixels


