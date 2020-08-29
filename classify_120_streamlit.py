import sys
#import cv2
import pickle
import numpy as np
import json
from tensorflow.keras.models import load_model,model_from_json
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import streamlit as st 
from keras.applications.imagenet_utils import decode_predictions

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], False)

def predict(image1): 
    #model = MobileNetV2(include_top=True, weights='imagenet')
    model = load_model('tl_fine_tuning_InceptionResNetV2_120_breeds.h5')
    #model = model_from_json('model.json')
    #with open('model.json', 'r') as json_file:
    #    model = model_from_json(json_file.read())
    image = load_img(image1, target_size=(299, 299))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((-1, image.shape[0], image.shape[1], 3)) / 255
    #img = cv2.imread(image1)
    #img = cv2.resize(img, (299, 299))
    #img = np.reshape(img, (-1, 299, 299, 3)) / 255
    with open('classes_encoding_120', 'rb') as f:
        classes_labels = pickle.load(f)
    label , probability = classes_labels[model.predict_classes(image)[0]], np.max(model.predict_proba(image))*100
    return label, probability        