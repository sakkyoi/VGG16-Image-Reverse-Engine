import gradio as gr

import numpy as np
import h5py
import pandas as pd
import pickle

from skimage.util import random_noise
import time

import tensorflow as tf

from model import VGGNet

def query_normal(query_img: np.ndarray):
    query_feature = model.extract_feature(query_img, verbose=0)
    scores = np.dot(query_feature, features.T)
    rank_ID = np.argsort(scores)[::-1]
    rank_score = scores[rank_ID]

    imlist = [image_ids.astype(str)[index] for i, index in enumerate(rank_ID[0:10])]

    return imlist, rank_score[0:10].tolist(), len(scores)

def query_kmeans(query_img: np.ndarray):
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

    return imlist, rank_score[0:10].tolist(), len(scores), cluster

def query(image, mode, noise, noise_seed, mean, var, amount, salt_vs_pepper):
    if image == None or noise_seed == None:
        return None, None, None, None, None
    
    query_img = tf.keras.utils.load_img(image, target_size=(model.input_shape[0], model.input_shape[1]))
    query_img = tf.keras.utils.img_to_array(query_img).astype(int)

    if noise == 'none':
        pass
    elif noise == 'gaussian' or noise =='speckle':
        query_img = random_noise(query_img / 255, mode=noise, rng=int(noise_seed), mean=mean, var=var, clip=True)
        query_img = np.array(query_img * 255, dtype=np.uint8)
    elif noise == 'localvar':
        query_img = random_noise(query_img / 255, mode=noise, rng=int(noise_seed), clip=True)
        query_img = np.array(query_img * 255, dtype=np.uint8)
    elif noise == 'poisson':
        query_img = random_noise(query_img / 255, mode=noise, rng=int(noise_seed), clip=True)
        query_img = np.array(query_img * 255, dtype=np.uint8)
    elif noise == 'salt' or noise == 'pepper':
        query_img = random_noise(query_img / 255, mode=noise, rng=int(noise_seed), amount=amount, clip=True)
        query_img = np.array(query_img * 255, dtype=np.uint8)
    elif noise == 's&p':
        query_img = random_noise(query_img / 255, mode=noise, rng=int(noise_seed), amount=amount, salt_vs_pepper=salt_vs_pepper, clip=True)
        query_img = np.array(query_img * 255, dtype=np.uint8)

    start = time.time()
    if mode == 'normal':
        results, scores, length = query_normal(query_img)
    elif mode == 'kmeans':
        results, scores, length, cluster = query_kmeans(query_img)
    end = time.time()
    query_time = end - start
    query_time = round(query_time * 1000, 2)

    return query_img, f'Query time: {query_time} ms', [(result, f'Score: {score}, file: {result}') for result, score in zip(results, scores)], length, f'{cluster}' if mode == 'kmeans' else None

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
            gr.Radio(['normal', 'kmeans'], value='normal'),
            gr.Radio(['none', 'gaussian', 'localvar', 'poisson', 'salt', 'pepper', 's&p', 'speckle'], value='none'),
            gr.Number(label='random_seed', value=0),
            gr.Slider(label='mean', maximum=1, minimum=0, value=0),
            gr.Slider(label='var', maximum=1, minimum=0, value=0.01),
            gr.Slider(label='amount', maximum=1, minimum=0, value=0.05),
            gr.Slider(label='salt_vs_pepper', maximum=1, minimum=0, value=0.5),
        ],
        outputs=[
            gr.Image(label='query image', type='numpy'),
            gr.Label(label='query time'),
            gr.Gallery(label='Top 10 similar images').style(columns=[5], rows=[2]),
            gr.Label(label='count of images for searching'),
            gr.Label(label='cluster id')
        ],
    )
    iface.launch()