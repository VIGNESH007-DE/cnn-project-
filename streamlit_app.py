import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# Reduce TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# App title
st.title("✍️ Handwritten Digit Recognition (CNN)")
st.write("Upload an image of a digit (0–9)")

# Load model safely
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_cnn_model.h5")

model = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        # Open and process image
        image = Image.open(uploaded_file).convert('L')
        image = image.resize((28, 28))

        # Show image
        st.image(image, caption="Uploaded Image", width=150)

        # Convert to array
        img_array = np.array(image) / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        # Prediction
        prediction = model.predict(img_array)
        result = np.argmax(prediction)

        # Confidence
        confidence = np.max(prediction)

        # Output
        st.success(f"Predicted Digit: {result}")
        st.info(f"Confidence: {confidence:.2f}")

    except Exception as e:
        st.error(f"Error processing image: {e}")

else:
    st.warning("Please upload an image to predict.")

# Footer
st.markdown("---")
st.write("Model: CNN trained on MNIST dataset (~98% accuracy)")
