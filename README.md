# COGS 260 Final Project: Biomedical Image Segmentation Using U-Net
### Authors: Lucas Tindall and Andrew Saad

## [Models](models)
+ contains the various U-Net model architectures we used for our project
+ our final model for the Ultrasound Nerve Dataset was
  [models/train_long.py](models/train_long.py)
+ our final model for the hippocampus mitochondria dataset was
  [models/train_long_mito.py](models/train_long_mito.py)

## [data.py](data.py)
+ used to convert raw image files into large .npy data files

## [Output texts](output_txts)
+ console outputs from our various training sessions

# Acknowledgement
+ This repository was based off of Marko Jocić U-Net keras implementation found
  at
[https://github.com/jocicmarko/ultrasound-nerve-segmentation](https://github.com/jocicmarko/ultrasound-nerve-segmentation)
