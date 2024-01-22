import streamlit as st
import keras
import numpy as np
import cv2

CLASSES = ["building", "forest", "glacier", "mountain", "sea", "street"]


def prediction_to_class(prediction):
    guesses = {}
    max_idx = 0
    for i in range(len(prediction)):
        if prediction[max_idx] < prediction[i]:
            max_idx = i
        if prediction[i] > 0.5:
            image_class = CLASSES[i]
            guesses[image_class] = prediction[i]

    max_class = CLASSES[max_idx]
    guesses[max_class] = prediction[max_idx]

    return guesses


def stringify_prediction(prediction):
    guesses = prediction_to_class(prediction)

    str_prediction = ""
    for key, value in guesses.items():
        str_prediction += f"{key} ({value * 100: .1f}%)  "

    return str_prediction


def prepare_image(image):
    image = cv2.resize(image, (128, 128))

    image = np.array(image)

    return image.reshape(1, 128, 128, 3) / 255.0


def run_ui():
    model: keras.models.Model = keras.models.load_model("trained_model")

    uploaded_image = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

    if not uploaded_image:
        return None

    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image)

    image = prepare_image(image)

    prediction = model.predict(image)[0]

    prediction = stringify_prediction(prediction)

    st.write(f"This is {prediction}")
