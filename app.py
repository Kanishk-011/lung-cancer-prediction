import streamlit as st

st.set_page_config(page_title="Lung Cancer Detection", layout="centered")

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

IMAGE_SIZE = 256

def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax'),
    ])

@st.cache_resource
def load_cnn_model():
    model = build_model()
    model.load_weights("LungCancerPrediction.h5")
    return model

model = load_cnn_model()
class_labels = ['lung_scc', 'lung_n', 'lung_acc']

st.title("🫁 Lung Cancer Detection using CNN")
st.markdown("Upload a chest X‑ray to predict lung cancer class.")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Chest X‑ray", use_column_width=True)

    if st.button("Predict"):
        img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        pred_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][pred_idx])

        st.success(f"**Predicted:** {class_labels[pred_idx]}")
        st.info(f"Confidence: {confidence * 100:.2f}%")
