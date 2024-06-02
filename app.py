import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load the model
model = load_model('my_model.keras')

st.title('Potato Disease Classification')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = plt.imread(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make prediction
    img_array = image.load_img(uploaded_file, target_size=(256, 256))
    img_array = image.img_to_array(img_array)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    class_names = ['Early Blight', 'Healthy', 'Late Blight']

    st.write("Prediction:")
    for i in range(len(class_names)):
        st.write(f"{class_names[i]}: {prediction[0][i]}")
