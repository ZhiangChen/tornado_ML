# Gaussian Process for Tornado Damage Mapping
Zhiang Chen, Sept 2020

## Processes
#### 1. rescale.py
It will resize the original tif files, and create resized detection and classification heatmaps. It also provides methods of recovering an image file to original tif file format.

#### 2. sampling.py
It generates samples for training (including test) data and also the data for inference demands. It can also stitch the samples and recover an image as the same size as the original. 

#### 3. gp.py
It provides methods of training a Gaussian Process model and doing inference with the model.
