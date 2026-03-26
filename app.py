import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import joblib
from PIL import Image

st.set_page_config(page_title="Handwritten Digit & Operator Classifier", page_icon="✍️", layout="centered")

MODEL_PATH = "best_handwritten_operator_model.keras"
LABEL_ENCODER_PATH = "label_encoder.pkl"
IMG_SIZE = 28

@st.cache_resource
def load_model_and_encoder():
    model = tf.keras.models.load_model(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return model, label_encoder

model, label_encoder = load_model_and_encoder()

def preprocess_uploaded_image(image, img_size=28):
    image = np.array(image)

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    image = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_NEAREST)

    # Invert if background is dark and digit is light
    if np.mean(image) > 127:
        image = 255 - image

    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=-1)   # (28,28,1)
    image = np.expand_dims(image, axis=0)    # (1,28,28,1)

    return image

st.title("✍️ Handwritten Digit & Operator Classifier")
st.write("Upload a handwritten digit or operator image, and the model will predict the class.")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    processed = preprocess_uploaded_image(image, IMG_SIZE)

    pred_probs = model.predict(processed)
    pred_idx = np.argmax(pred_probs, axis=1)[0]
    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    confidence = float(np.max(pred_probs)) * 100

    st.success(f"Prediction: {pred_label}")
    st.info(f"Confidence: {confidence:.2f}%")

    st.subheader("Prediction Probabilities")
    probs_dict = {
        label_encoder.inverse_transform([i])[0]: float(pred_probs[0][i]) * 100
        for i in range(len(pred_probs[0]))
    }

    sorted_probs = dict(sorted(probs_dict.items(), key=lambda x: x[1], reverse=True))
    st.bar_chart(sorted_probs)
