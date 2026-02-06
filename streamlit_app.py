import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

model = tf.keras.models.load_model(
    "person_classifier.h5",
    compile=False
)

IMG_SIZE = 224

# Function predict
def predict(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)[0][0]
    if prediction > 0.5:
        return "Person", prediction
    else:
        return "Not Person", 1 - prediction

# Streamlit UI
st.title("Person vs Not Person Classifier")
st.write("Upload an image to classify if it contains a person or not.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label, prob = predict(img)
    st.write(f"Prediction: **{label}** (Probability: {prob:.2f})")
