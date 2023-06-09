# VGG16-Image-Reverse-Engine
An Image Reverse Engine based on VGG16

## Introduction
This is a project for the course "Application of artificial intelligence and deep learning" in the National Kaohsiung University of Science and Technology. The goal of this project is to build an image reverse engine based on VGG16. 

## Requirements
```
pip install -r requirements.txt
```

## Usage

### [main.ipynb](main.ipynb)
The main.ipynb is the main file of this project. You can run it on Jupyter Notebook or Google Colab.
It contains the following parts:
1. Load data
2. Load VGG16 model
3. Indexing images
4. Save features and indexes
5. Load features and indexes
6. Search images
7. Show results

> The dataset we used is from [Stanford University CS231n](http://cs231n.stanford.edu/tiny-imagenet-200.zip). You can download it and put it in the same directory as the main.ipynb.

### [query.ipynb](query.ipynb)
The query.ipynb is a file for you to query images. You can run it on Jupyter Notebook or Google Colab.
It contains the following parts:
1. Search images without any modification
2. Search images with Gaussian Noise
3. Search images with clustering (K-Means) and Gaussian Noise

This file also explains the process of searching images in detail.

### [indexing.py](indexing.py)
The indexing.py is a file for you to index images. It is separated from the index part in main.ipynb.

### [model.py](model.py)
The model.py is a file for you to load VGG16 model. You can use it to load VGG16 model in other files.
```python
from model import VGGNet
model = VGGNet()
```
and then you can use the model to extract features.
```python
features = model.extract_feature(image)
```

### [web_app.py](web_app.py)
The web_app.py is a file for you to run a web application built by Gradio. You can run it on your local machine.
```
python web_app.py
```

## Reference
[Tiny Imagenet 200 from Stanford University CS231n](http://cs231n.stanford.edu/tiny-imagenet-200.zip)

[how does the dot product determine similarity?](https://math.stackexchange.com/questions/689022/how-does-the-dot-product-determine-similarity)

[Image Similarity Comparison using VGG16 Deep Learning Model](https://medium.com/@developerRegmi/image-similarity-comparison-using-vgg16-deep-learning-model-a663a411cd24)

[基于VGG-16的海量图像检索系统（以图搜图升级版）](https://www.cnblogs.com/linkmust/articles/9607604.html)

[点积相似度、余弦相似度、欧几里得相似度](https://zhuanlan.zhihu.com/p/159244903)