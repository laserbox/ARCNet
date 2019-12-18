import os
import numpy as np


def img_path_generator(dataset='ucm'):
    if not os.path.exists('datasets'):
        os.makedirs('datasets')
    img_dir = 'datasets/%s/' % dataset
    img_path = []
    img_labels = []
    dicts = os.listdir(img_dir)
    for root, _, files in os.walk(img_dir):
        for name in files:
            img_path.append(os.path.join(root, name))
            label_name = root.split('/')[-1]
            img_labels.append(int(dicts.index(label_name)))
    return np.array(img_path, dtype=object), np.array(img_labels), len(dicts)
