# Gaussian Process for Tornado Damage Mapping
Zhiang Chen, Sept 2020

## Processes
#### 1. rescale.py
It will resize the original tif files, and create resized detection and classification heatmaps. It also provides methods of recovering an image file to original tif file format.

#### 2. sampling.py
It generates samples for training (including test) data and also the data for inference demands. It can also stitch the samples and recover an image as the same size as the original. 

#### 3. training_module.py
It includes sampling, training, and inference for training data. 

#### 4. groundtruth_module.py
It includes sampling, training, and inference for training data

To implement GP regression in batch,
```buildoutcfg
bash train_n_groundtruth.bash
```

#### 5. average_results.py
It implements Monte Carlo sampling. 

#### 6. semivariogram.py
It draws semivariogram in a training dataset.
