import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("mnist_cnn_model.h5")

st.title("✍️ Handwritten Digit Recognition (CNN)")

st.write("Upload a digit image (0–9)")

uploaded_file = st.file_uploader("Choose an image", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    image = image.resize((28,28))

    st.image(image, caption="Uploaded Image", width=150)

    img_array = np.array(image)/255.0
    img_array = img_array.reshape(1,28,28,1)

    prediction = model.predict(img_array)
    result = np.argmax(prediction)

    st.success(f"Predicted Digit: {result}")
