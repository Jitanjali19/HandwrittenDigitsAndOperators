import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import json

st.set_page_config(page_title="Handwritten Digits and Operators", layout="centered")

st.title("Handwritten Digits and Operators Recognition")
st.write("Upload an image to predict the handwritten digit/operator.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("handwritten_model.h5")

@st.cache_data
def load_label_map():
    with open("label_map.json", "r") as f:
        return json.load(f)

model = load_model()
label_map = load_label_map()

uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

def preprocess_image(image):
    image = image.convert("L")          # grayscale
    image = ImageOps.invert(image)      # agar background white aur digit dark ho to useful
    image = image.resize((28, 28))      # training size ke hisab se
    img = np.array(image)

    img = img.astype("float32") / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", width=180)

    x = preprocess_image(image)
    pred = model.predict(x)
    pred_idx = int(np.argmax(pred, axis=1)[0])
    confidence = float(np.max(pred) * 100)

    predicted_label = label_map.get(str(pred_idx), str(pred_idx))

    st.subheader(f"Prediction: {predicted_label}")
    st.write(f"Confidence: {confidence:.2f}%")
