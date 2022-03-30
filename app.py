# Import warnings 
from urllib import request
import warnings
# Stopping all warning messages
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

# Stop debug logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Importing all required libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import load_img, img_to_array, array_to_img
import matplotlib.pyplot as plt
import streamlit as st
import requests
from requests import exceptions
import numpy as np
from io import BytesIO, TextIOWrapper
from PIL import Image
import re


st.title("Bone Class Image Classifier")
st.text("Please provide the bone url for bone class image classification.")

# Classes for our prediction
classification_classes = ["Healthy_bones", "Fractured_bones", "Bones_beyond_repair"]
classification_classes_dict = {0:"Healthy Bones", 1:"Fractured Bones", 2:"Bones Needing Prolong treatment"}

EXCEPTIONS = {IOError, FileNotFoundError, exceptions.RequestException, exceptions.HTTPError, exceptions.ConnectionError,
              exceptions.Timeout}


@st.cache(allow_output_mutation=True)
def load_model():
    cnn_model = tf.keras.models.load_model("Models/Transfer_learning_model_v1.h5", compile=False)
    return cnn_model

with st.spinner("Loading Model Into Memory..........."):
    cnn_model = load_model()
    

# Function file for getting image data
def get_image_data(bone_image, image_resize_value):

  # Set image size
  img_size = image_resize_value

  # Processing image
  try:
    #st.text(bone_image)
    x_ray_image = tf.image.decode_jpeg(bone_image, channels=3)
    x_ray_image = tf.cast(x_ray_image, tf.float32)
    x_ray_image = x_ray_image/255
    x_ray_image = tf.image.resize(x_ray_image, [img_size, img_size])
    x_ray_image = np.expand_dims(x_ray_image, axis=0)
  except:
    print("Some error occured in fetching data!")  
    
  return x_ray_image  


# Function to predict image class
def predict_image_class(bone_image):
    image_data = get_image_data(bone_image, image_resize_value=224)
    try:
        bone_class_image = Image.open(BytesIO(bone_image))
    except:
        bone_class_image = Image.open(bone_image)
    class_probabilities = cnn_model.predict(image_data)
    bone_image_class = classification_classes_dict[np.argmax(class_probabilities)]
    bone_image_class_probability = round(100*class_probabilities[0][np.argmax(class_probabilities)], 5)
    
    return bone_class_image, bone_image_class, bone_image_class_probability


# Main program 
default_bone_image_path = "https://prod-images-static.radiopaedia.org/images/25293630/3a9c605b616de1ed6dd1d2397bc394_jumbo.jpeg"
image_path = st.text_input("Enter Image URL to classify:", default_bone_image_path)
if image_path is not None:
    try:
        st.text("[INFO] fetching: {}".format(image_path))
        content = requests.get(image_path).content
        st.write("Predicted Bone class: ")
        with st.spinner("Classifying........."):
            bone_class_image, bone_image_class, bone_image_class_probability = predict_image_class(content)
            st.write(f"The predicted bone class is {bone_image_class} with probability {bone_image_class_probability}%")
        st.write("")
        st.write("The image requested for classification is: ")
        st.image(bone_class_image, caption="Classifying Bone Image", use_column_width=True)        
    except Exception as e:
        st.text(e)
        if type(e) in EXCEPTIONS:
            st.text("Error occured in loading image file")
            st.text("[INFO] fetching: {}".format(image_path))
    
    
    



