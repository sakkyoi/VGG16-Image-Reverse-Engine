from fastapi import FastAPI, File, UploadFile
import uvicorn

from PIL import Image
from io import BytesIO
import base64
import numpy as np
import h5py
import pandas as pd
from sklearn.cluster import KMeans
import pickle

import tensorflow as tf

from model import VGGNet

def query(image: BytesIO):
    query_img = tf.keras.utils.load_img(image, target_size=(model.input_shape[0], model.input_shape[1]))
    query_img = tf.keras.utils.img_to_array(query_img).astype(int)

    query_feature = model.extract_feature(query_img, verbose=0)
    scores = np.dot(query_feature, features.T)
    rank_ID = np.argsort(scores)[::-1]
    rank_score = scores[rank_ID]

    imlist = [image_ids.astype(str)[index] for i, index in enumerate(rank_ID[0:10])]

    results = []
    for i, im in enumerate(imlist):
        buffered = BytesIO()
        Image.open(im).save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())
        img_str = 'data:image/jpeg;base64,' + img_str.decode('utf-8')

        results.append(img_str)

    return results, rank_score[0:10].tolist()

def query_kmeans(image: BytesIO):
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

    results = []
    for i, im in enumerate(imlist):
        buffered = BytesIO()
        Image.open(im).save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())
        img_str = 'data:image/jpeg;base64,' + img_str.decode('utf-8')

        results.append(img_str)

    return results, rank_score[0:10].tolist()

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/search")
async def search(file: UploadFile = File(...)):
    imlist, rank_score = query(BytesIO(file.file.read()))
    return {"images": imlist, "scores": rank_score}

@app.post("/search_kmeans")
async def search_kmeans(file: UploadFile = File(...)):
    imlist, rank_score = query_kmeans(BytesIO(file.file.read()))
    return {"images": imlist, "scores": rank_score}

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

    uvicorn.run(app, host='0.0.0.0', port=8000)