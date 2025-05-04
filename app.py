import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("satellite_cnn_model.h5")

# Define the classes (same as during training)
classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 
           'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

# Prediction function
def predict_image(uploaded_image):
    image = Image.open(uploaded_image).convert('RGB')
    image = image.resize((64, 64))
    img_array = np.array(image, dtype='float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    class_name = classes[class_index]
    confidence = prediction[0][class_index]

    return class_name, confidence, image

# Streamlit UI
st.set_page_config(page_title="Satellite Image Classifier")
st.title("🛰️ Satellite Image Classification")
st.write("Upload a satellite image to predict its land cover type.")

uploaded_file = st.file_uploader("Choose a satellite image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    class_name, confidence, img = predict_image(uploaded_file)

    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.markdown(f"### 🧠 Predicted Class: **{class_name}**")
    st.markdown(f"### 🔍 Confidence: **{confidence:.2f}**")

