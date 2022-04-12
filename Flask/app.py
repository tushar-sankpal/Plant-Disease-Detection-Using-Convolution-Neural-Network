
from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as nps
#import tensorflow as tf
import tensorflow as tf


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'plant.h5' 

# Load your trained model
model = load_model(MODEL_PATH)




def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(256, 256))

    # Preprocessing the image
    x = image.img_to_array(img) # [25*256]
    ## x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = nps.expand_dims(x, axis=0)  # np 
   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    # x = preprocess_input(x)

    preds = model.predict(x)
    preds=nps.argmax(preds, axis=1) # np
    if preds==0:
        preds="Plant Name : Bell Pepper | Disease : Bacterial Spot"
    elif preds==1:
        preds="Plant Name : Bell Pepper | No Disease : Healthy"
    elif preds==2:
        preds="Plant Name : Potato | Disease : Early Blight"
    elif preds==3:
        preds="Plant Name : Potato | Disease : Late Blight"
    elif preds==4:
        preds="Plant Name : Potato | No Disease : Healthy"
    elif preds==5:
        preds="Plant Name : Tomato | Disease : Bacterial Spot"
    elif preds==6:
        preds="Plant Name : Tomato | Disease : Early Blight"
    elif preds==7:
        preds="Plant Name : Tomato | Disease : Late Blight"
    elif preds==8:
        preds="Plant Name : Tomato | Disease : Leaf Mold"
    elif preds==9:
        preds="Plant Name : Tomato | Disease : Septoria Leaf Spot"
    elif preds==10:
        preds="Plant Name : Tomato | Disease : Spider Mites Two Spotted Spider Mite"
    elif preds==12:
        preds="Plant Name : Tomato | Disease : Target Spot"
    elif preds==13:
        preds="Plant Name : Tomato | Disease : Yellow Leaf Curl Virus"
    elif preds==14:
        preds="Plant Name : Tomato | Disease : Mosaic Virus"
    else:
        preds ="Plant Name : Tomato | No Disease : Healthy"
    
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file'] # xyz 

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename)) #plant/uploads/xyz.jpg
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(port=5001,debug=True)

