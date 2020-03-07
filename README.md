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
## train the baseline model
```
    python baseline_train.py
```
## train the arcnet model:
```
    python train.py
```