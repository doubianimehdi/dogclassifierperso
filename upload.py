import streamlit as st 
from PIL import Image
from classify_120_streamlit import load_model,preprocess,load_classeslabels,to_predictions_chart,top5_proba
import time
import io
import numpy as np
import pickle
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array
import tensorflow as tf



st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Dog Breed Classifier")

with st.spinner('Loading Model....'):
  model = load_model()
  
classes_labels = load_classeslabels()  
  
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

if file is None:
  st.text("You haven't uploaded an image file")
else:
  original_img = Image.open(file)
  st.image(original_img, use_column_width=True)
  img = image.img_to_array(original_img)
  a = preprocess(img)
  probs = model.predict(a)
  pred = classes_labels[np.argmax(probs)] 
  st.text("Breed")
  st.write(pred)
  st.text("Probability") 
  st.write(np.max(probs)*100)  
  st.text("Top 5 probability")
  top5proba = top5_proba(probs,classes_labels)
  st.write(top5proba)
  st.write(to_predictions_chart(top5proba))    