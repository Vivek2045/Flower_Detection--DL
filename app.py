import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Load the trained model
model = load_model('Model.h5')

# List of class labels (you should modify these based on the labels used in your dataset)
class_labels = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']

# Streamlit page configuration
st.set_page_config(page_title="Flower Detection", page_icon="🌸")

# Title of the app
st.title("Flower Detection App")

# Instructions for the user
st.markdown("Upload a flower image to detect its type.")

# Upload the image
uploaded_image = st.file_uploader("Choose a flower image...", type=["jpg", "jpeg", "png"])

# If an image is uploaded, perform prediction
if uploaded_image is not None:
    # Display the image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = image.load_img(uploaded_image, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image

    # Predict the class of the uploaded image
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]

    # Show the result
    st.write(f"Prediction: **{predicted_class}**")
    st.write(f"Confidence: {np.max(prediction) * 100:.2f}%")

