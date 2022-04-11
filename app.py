# Import warnings 
from urllib import request
import warnings
import os


# Importing all required libraries
import tensorflow as tf
from tensorflow import keras
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
#Add sidebar to the app
st.sidebar.markdown("### Welcome to my Bone Classifier Web App")
st.sidebar.markdown("Hi! I am Deepankar. This web app serves purpose of classifying bones into three major categories. This web app is developed keeping in mind for faster screening of medical records for taking further necessary steps. Working in backend is a computer vision deep learning algorithm for serving your purpose. Enjoy!")
st.markdown("#### There are three class for prediction:")
st.markdown("- Healthy Bones")
st.markdown("- Fractured Bones")
st.markdown("- Bones Needing Prolong Treatment Or Beyond Repair")
st.markdown("#### Glimpses of each class:")

# Images
col1, col2, col3 = st.columns(3)

healthy_img = Image.open("Images/healthy_bones_show.jpg")
#col1.header("Healthy Bones")
col1.image(healthy_img, use_column_width=True)

fractured_img = Image.open("Images/fractured_bones_show.jpg")
#col2.header("Fractured Bones")
col2.image(fractured_img, width=225)

beyond_repair_img = Image.open("Images/beyond_repair_bones_show.jpg")
#col2.header("Bones Requiring Prolong treatment Or Beyond Repair")
col3.image(beyond_repair_img, width=150)

st.markdown("#### Categories of bones inside each class and types of bones on which Deep Learning model has been trained on:")
categories_img = Image.open("Images/categories.jpg")
st.image(categories_img, caption="Information on each class", use_column_width=True)


st.markdown("#### Please provide the x-ray bone image url for bone class image classification.")
st.text("Please provide x-ray bone image url from the above mentioned bone categories and types of bones only for prediction.")

# Classes for our prediction
classification_classes = ["Healthy_bones", "Fractured_bones", "Bones_beyond_repair"]
classification_classes_dict = {0:"Healthy Bones", 1:"Fractured Bones", 2:"Bones Requiring Prolong treatment Or Beyond Repair"}

EXCEPTIONS = {IOError, FileNotFoundError, exceptions.RequestException, exceptions.HTTPError, exceptions.ConnectionError,
              exceptions.Timeout}


@st.cache(allow_output_mutation=True)
def load_model():
    cnn_model = tf.keras.models.load_model("Models/Transfer_learning_model_v3_M1.h5", compile=False)
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
    print("Some error occurred in fetching data!")  
    
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
default_bone_image_path = "https://www.healthpages.org/wp-content/uploads/hand-x-ray.jpg"
try:
    image_path = st.text_input("Enter Image URL to classify:", default_bone_image_path)
except:
    print("Image output not found from the input url! Try again or enter another url.")
    
if image_path is not None:
    try:
        st.text("[INFO] fetching: {}".format(image_path))
        content = requests.get(image_path).content
        st.write("Predicted Bone class: ")
        with st.spinner("Classifying........."):
            bone_class_image, bone_image_class, bone_image_class_probability = predict_image_class(content)
            if bone_image_class_probability > 70:
                st.success(bone_image_class)
            elif bone_image_class_probability < 70 and bone_image_class_probability >= 40:
                st.warning(bone_image_class)
            else:
                st.error(bone_image_class)
            st.write(f"The predicted probability: {bone_image_class_probability}%")
        st.write("")
        st.write("The image requested for classification is: ")
        st.image(bone_class_image, caption="Classifying Bone Image", use_column_width=True)        
    except Exception as e:
        st.text(e)
        if type(e) in EXCEPTIONS:
            st.text("Error occurred in loading image file")
            st.text("[INFO] fetching: {}".format(image_path))
    
    
    



