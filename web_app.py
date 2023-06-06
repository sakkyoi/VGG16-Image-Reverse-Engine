import gradio as gr

import numpy as np
import h5py
import pandas as pd
import pickle

import tensorflow as tf

from model import VGGNet

def query_normal(image: str):
    query_img = tf.keras.utils.load_img(image, target_size=(model.input_shape[0], model.input_shape[1]))
    query_img = tf.keras.utils.img_to_array(query_img).astype(int)

    query_feature = model.extract_feature(query_img, verbose=0)
    scores = np.dot(query_feature, features.T)
    rank_ID = np.argsort(scores)[::-1]
    rank_score = scores[rank_ID]

    imlist = [image_ids.astype(str)[index] for i, index in enumerate(rank_ID[0:10])]

    return imlist, rank_score[0:10].tolist()

def query_kmeans(image: str):
    query_img = tf.keras.utils.load_img(image, target_size=(model.input_shape[0], model.input_shape[1]))
    query_img = tf.keras.utils.img_to_array(query_img).astype(int)

    query_feature = model.extract_feature(query_img, verbose=0)

    cluster = kmeans.predict(query_feature.reshape(1, -1))
    cluster = cluster[0]

    df = pd.DataFrame({'image_id': image_ids.astype(str), 'cluster_id': kmeans.labels_})
    df = df[df['cluster_id'] == cluster]

    query = df[df['cluster_id'] == cluster].index

    query_feature = model.extract_feature(query_img)
    scores = np.dot(query_feature, features[query].T)
    rank_ID = np.argsort(scores)[::-1]
    rank_score = scores[rank_ID]

    imlist = [image_ids.astype(str)[index] for i, index in enumerate(query[rank_ID[0:10]])]

    return imlist, rank_score[0:10].tolist()

def query(image, type):
    if type == 'kmeans':
        results, scores = query_kmeans(image)
    elif type == 'normal':
        results, scores = query_normal(image)

    return [(result, f'Score: {score}, file: {result}') for result, score in zip(results, scores)]

if __name__ == '__main__':
    model = VGGNet()

    # Load dataset
    datasets = h5py.File('features.h5', 'r')
    features = datasets['features'][:]
    image_ids = datasets['image_ids'][:]
    datasets.close()

    # Load kmeans model
    with open('kmeans.pkl', 'rb') as f:
        kmeans = pickle.load(f)

    # Run web app
    iface = gr.Interface(
        title='An image search engine based on VGG16',
        fn=query,
        inputs=[
            gr.Image(type='filepath'),
            gr.Radio(['normal', 'kmeans'])
        ],
        outputs=[
            gr.Gallery(label='Top 10 similar images', show_label=False).style(columns=[5], rows=[2])
        ],
    )
    iface.launch()