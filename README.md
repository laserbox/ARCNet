# ARCNet
## requirement
pytorch>1.0

python=3.6
## symbolic link the dataset to the datasets folder:
```
datasets/
        ucm/
            class_1/
            class_2/
            ...
        whu/
            class_1/
            class_2/
            ...
        aid/
            class_1/
            class_2/
            ...
        opt/
            class_1/
            class_2/
            ...
```
## first: train the baseline model, fine-tune the imagenet pretrained model on remote sensing dataset:
```
    python baseline_train.py
```
## second: train the arcnet model:
```
    python train.py
```
