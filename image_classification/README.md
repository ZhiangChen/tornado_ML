# Image Classification
Image classification for tornado damage estimation.

## Training
The training is done by [training.py](./training.py). A neural network architecture must be specified to initialize a neural network: `model = neural_network('resnext101_32x8d', nm_classes)`. The available models can be listed by the following script:
```buildoutcfg
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
```

## Validation
[valid_densenet161.py](./valid_densenet161.py)  
[valid_densenet161_binary.py](./valid_densenet161_binary.py)  
[valid_densenet201.py](./valid_densenet201.py)  
[valid_densenet201_binary.py](./valid_densenet201_binary.py)  
[valid_resnet152.py](./valid_resnet152.py)  
[valid_resnet152_binary.py](./valid_resnet152_binary.py)  
[valid_resnext101.py](./valid_resnext101.py)  
[valid_resnet101_binary.py](./valid_resnet101_binary.py)  

## Test inference
The test inference is done by [infer.py](./infer.py) and [infer_binary.py](./infer_binary.py).



