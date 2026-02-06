import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
MODEL_FILENAME = "person_classifier.h5"
IMG_SIZE = 224

@st.cache_resource
def load_model(path):
    try:
        return tf.keras.models.load_model(path)
    except Exception:
        return None

# Resolve model path relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)

model = load_model(MODEL_PATH)

# Function predict
def predict(img, model):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)[0][0]
    if prediction > 0.5:
        return "Person", float(prediction)
    else:
        return "Not Person", float(1 - prediction)

# Streamlit UI
st.title("Person vs Not Person Classifier")
st.write("Upload an image to classify if it contains a person or not.")

if model is None:
    st.error("Model file not found or failed to load. Ensure 'person_classifier.h5' is in the repository root.")
else:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        label, prob = predict(img, model)
        st.write(f"Prediction: **{label}** (Probability: {prob:.2f})")