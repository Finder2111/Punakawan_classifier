import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image


# Input saved model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("./vgg16_small_20.h5")


# Make prediction
def classify_image(model, image):
    # Resize and preprocess the image according to VGG16 requirements
    image = image.resize((175, 225))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.vgg16.preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    # Perform prediction
    prediction = model.predict(image)

    # Get the class label with the highest probability
    class_label = np.argmax(prediction[0])
    return class_label


# Image uploader
def main():
    st.title("Klasifikasi Citra Wayang Punakawan")
    st.write("Silakan upload gambar Punakawan yang ingin kamu kenali")

    # Load the model
    model = load_model()

    # Create a file uploader to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Hasil upload", use_column_width=True)

        # Classify the image
        class_label = classify_image(model, image)
        class_names = ["Bagong", "Petruk", "Semar", "Gareng"]
        st.write("Hasil Prediksi:", class_names[class_label])


if __name__ == "__main__":
    main()
