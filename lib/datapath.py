import os


ucmdicts = {'forest': 0, 'buildings': 1, 'river': 2, 'mobilehomepark': 3, 'harbor': 4, 'golfcourse': 5,
            'agricultural': 6, 'runway': 7, 'baseballdiamond': 8, 'overpass': 9, 'chaparral': 10, 'tenniscourt': 11,
            'intersection': 12, 'airplane': 13, 'parkinglot': 14, 'sparseresidential': 15, 'mediumresidential': 16,
            'denseresidential': 17, 'beach': 18, 'freeway': 19, 'storagetanks': 20}


def img_path_generator(dataset='ucm'):
    img_dir = 'datasets/%s/images/' % dataset
    img_path = []
    img_labels = []
    for root, _, files in os.walk(img_dir):
        for name in files:
            img_path.append(os.path(root, name))
            label_name = root.split('/')[-1]
            img_labels.append(eval(dataset+'dicts')[label_name])
    return img_path, img_labels, len(eval(dataset+'dicts'))
