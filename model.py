import tensorflow as tf

import numpy as np
from numpy import linalg as LA

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

class VGGNet:
    def __init__(self):
        self.input_shape = (224, 224, 3)
        self.weight = 'imagenet'
        self.pooling = 'max'
        self.model = VGG16(weights=self.weight, input_shape=(self.input_shape[0], self.input_shape[1], self.input_shape[2]), pooling=self.pooling, include_top=False)
        self.model.predict(np.zeros((1, 224, 224, 3)))

    def extract_feature(self, img_path: any, verbose='auto'):
        if type(img_path) == str or type(img_path) == np.str_:
            img = tf.keras.utils.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
            img = tf.keras.utils.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            feature = self.model.predict(img, verbose=verbose)
            feature_normalized = feature[0] / LA.norm(feature[0])
        elif type(img_path) == np.ndarray:
            img = img_path
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            feature = self.model.predict(img, verbose=verbose)
            feature_normalized = feature[0] / LA.norm(feature[0])
        else:
            raise TypeError('img_path must be str or np.ndarray')
        
        return feature_normalized