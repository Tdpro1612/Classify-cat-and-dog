# Classify-cat-and-dog

![](https://storage.googleapis.com/kaggle-competitions/kaggle/3362/media/woof_meow.jpg)

Welcome to the weekly project! Now it's the time to build your first ML application.

The idea is simple: "Cats vs Dogs" classification. Let's build a Flask app to help user upload their image and get the result that is a cat or dog photo.

You will see that not every time your model predicts correctly. Giving users a way to correct the answer if it's wrong or confirm if it's right will be very helpful to improve your model (make it smarter). 

Good luck, and have fun!


LINK TO DATASET: https://www.kaggle.com/tongpython/cat-and-dog

#import library
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import os
import pathlib
import IPython.display as display
import cv2
import time
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout,Activation
import random
```
# Load data,preprocessing image
*load data,clean data same 
```
Learningrate = 0.001
TRAIN_DIR = '/path upload data/cat_dog/train'
MODEL_NAME = 'Catanddog_Detect'
Img_size = 224
from os import listdir
from os.path import isfile, join
a = os.listdir(TRAIN_DIR)
image_paths = [os.path.join(TRAIN_DIR, f) for f in a]
import random
all_paths = random.sample(image_paths,len(image_paths))
all_paths = all_paths[:25000]
```
*Preprocessing image
```
# Preprocess an image
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [Img_size, Img_size])
    image /= 255.0
    return image

# Read the image from path and preprocess
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)
```
![preprocessingimage](https://user-images.githubusercontent.com/61773507/96670199-ff9ea880-1388-11eb-91e0-7b2557267951.png)
# Train model
```
model_history = model.fit(train_dataset, epochs = 10, validation_data = val_dataset)
```
![train](https://user-images.githubusercontent.com/61773507/96669857-4c35b400-1388-11eb-976b-9f6ad4c36c9c.png)

# Save model
```
model.save('Catanddog_Detect.h5')
```

# build flask to check

**after you run the file jupyter notebook you have the model Catanddog_Detect.h5,you coppy this,and create some forder like the tree**
```
* simple_flask
    * static
        * images
        * script.js
        * style.css
    * templates
        * classify.html
        * home.html
    * uploads
    * app.py
    * Catanddog_Detect.h5
```

You fix the path where you create the forder simple_flask,then run the app.py to show the flask

![home](https://user-images.githubusercontent.com/61773507/139358763-aa7de629-324f-48be-a3db-a8c0dd1ad1e4.jpg)

then you chose an image and click the button classify,this predict the image and show the result

![classify](https://user-images.githubusercontent.com/61773507/139358953-d913b7ab-9f0d-431a-ab3e-d7420ca17c17.jpg)

