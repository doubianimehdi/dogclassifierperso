import sys
#import cv2
import pickle
import numpy as np
import json
from tensorflow.keras.models import load_model
import streamlit as st 
from keras.applications.imagenet_utils import decode_predictions
from PIL import Image 
import requests
import altair as alt
import pandas as pd
from io import BytesIO
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], False)

@st.cache(allow_output_mutation=True)
def load_model():
  model = tf.keras.models.load_model('tl_fine_tuning_InceptionResNetV2_120_breeds')
  return model
  
@st.cache(allow_output_mutation=True)  
def load_classeslabels():
    with open('classes_encoding_120', 'rb') as f:
        classes_labels = pickle.load(f)
    return classes_labels
  
def preprocess(img):
  width, height = img.shape[0], img.shape[1]
  img = image.array_to_img(img, scale=False)
  img = img.resize((299,299))
  img = image.img_to_array(img)
  img /= 255.0
  img = np.expand_dims(img,axis=0)
  return img    
 
def to_predictions_chart(sortedproba) -> alt.Chart:
    chart = (alt.Chart(sortedproba[:5]).mark_bar().encode(x=alt.X("probability:Q", scale=alt.Scale(domain=(0, 100))),y=alt.Y("dog breed name:N",sort=alt.EncodingSortField(field="probability", order="descending"))))
    return chart

def top5_proba(probs,classes_labels) -> alt.Chart:
    labels = pd.DataFrame(classes_labels,columns=["dog breed name"]) 
    probability = pd.DataFrame(probs.transpose(),columns=["probability"])
    breedproba = pd.concat([labels, probability], axis=1, join='inner')
    sorted_breedproba = breedproba.sort_values(by='probability', ascending=False)
    sorted_breedproba["probability"] = sorted_breedproba["probability"].round(2) * 100
    sorted_breedtop5proba = sorted_breedproba[:5]
    return sorted_breedtop5proba
