import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model('my_model.keras')

class_names = ['Early_blight', 'Healthy', 'Late_blight']

# Function to make a prediction
def prediction(img):
    # Load the image and convert it to RGB if it's grayscale
    my_image = image.load_img(img, target_size=(256, 256))
    my_image = image.img_to_array(my_image)
    if my_image.shape[-1] == 1:  # If the image has only one channel (grayscale)
        my_image = np.concatenate([my_image] * 3, axis=-1)  # Convert to RGB

    my_image = np.expand_dims(my_image, 0)

    out = np.round(model.predict(my_image)[0], 2)
    return out

# Streamlit app UI
st.title("Potato Disease Classification")
st.write("Upload an image of a potato leaf to classify its disease status.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image_data = uploaded_file.read()
    st.image(image_data, caption='Uploaded Image', use_column_width=True)

    # Make prediction
    with open("temp_image.jpg", "wb") as f:
        f.write(image_data)

    prediction_result = prediction("temp_image.jpg")

    # Display prediction result
    fig, ax = plt.subplots()
    ax.barh(class_names, prediction_result, color='lightgray', edgecolor='red', linewidth=1, height=0.5)
    for index, value in enumerate(prediction_result):
        ax.text(value/2 + 0.1, index, f"{100*value:.2f}%", fontweight='bold')
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(class_names, fontweight='bold', fontsize=14)
    ax.set_xticks([])
    st.pyplot(fig)
