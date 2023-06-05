import tensorflow as tf

import os
import random
import gc

import numpy as np

import h5py

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from model import VGGNet

'''
    You have to download the dataset from [Stanford University](http://cs231n.stanford.edu/tiny-imagenet-200.zip) and unzip it to the same directory as this notebook.

    If you want to use your own dataset, you can modify the code below.

    The datas must a list of images' path like this:
    ```
    ['data\\image0.jpg', 'data\\image1.jpg', 'data\\image2.jpg', ...]
    ```
'''
def load_data():
    directories = os.listdir(os.path.join('tiny-imagenet-200', 'train'))
    datas = []
    for directory in directories:
        datas += [os.path.join('tiny-imagenet-200', 'train', directory, 'images', file) for file in os.listdir(os.path.join('tiny-imagenet-200', 'train', directory, 'images')) if file.endswith('.JPEG')]
    print(f'Successfully load {len(datas)} images')
    return datas

def indexing(datas):
    features = []
    image_ids = []

    model = VGGNet()
    for i, img_path in enumerate(datas):
        feature_normalized = model.extract_feature(img_path, verbose=0)
        features.append(feature_normalized)
        image_ids.append(img_path.encode())
        print(f'Extracted feature from {i+1} images, {len(datas) - i - 1} images left')

    return features, image_ids

if __name__ == '__main__':
    datas = load_data()

    features, image_ids = indexing(datas)

    print('Done')

    features = np.array(features)
    image_ids = np.array(image_ids)

    output = h5py.File('features.h5', 'w')
    output.create_dataset('features', data=features)
    output.create_dataset('image_ids', data=image_ids)
    output.close()