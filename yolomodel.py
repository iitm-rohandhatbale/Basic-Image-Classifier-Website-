from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from darkflow.net.build import TFNet

from keras.models import load_model
from keras.preprocessing import image

options = {"model": "cfg/yolo.cfg",
           "load": "bin/yolo.weights",
           "threshold": 0.5
           }

def boxing(original_img, predictions):
    newImage = np.copy(original_img)

    for result in predictions:
        top_x = result['topleft']['x']
        top_y = result['topleft']['y']

        btm_x = result['bottomright']['x']
        btm_y = result['bottomright']['y']

        confidence = result['confidence']
        label = result['label'] + " " + str(round(confidence, 3))

        if confidence > 0.3:
            newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (255,0,0), 3)
            newImage = cv2.putText(newImage, label, (top_x, top_y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.8, (0, 230, 0), 1, cv2.LINE_AA)

    return newImage

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/your_model.h5'

#Load your model
ymodel = TFNet(options)
print('Model loaded. Check http://127.0.0.1:5000/')
def model_predict(img_path, model):
    img = image.load_img(img_path)

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    #x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x, mode='caffe')

    preds = model.return_predict(x)
    return preds

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        img = Image.open(file_path)
        img_arr = np.asarray(img)
        preds = model_predict(file_path, ymodel)
        _, ax = plt.subplots(figsize=(20, 10))
        result = boxing(img_arr, preds)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)
        prediction_class = preds[0]['label']
        res = str(prediction_class)
             # Convert to string
        return res
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
