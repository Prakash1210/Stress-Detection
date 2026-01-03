import streamlit as st
import cv2
import numpy as np
from keras.models import model_from_json

# Load the emotion recognition model and face cascade
json_file = open('fer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("fer.h5")

# Setting image resizing parameters

WIDTH = 48
HEIGHT = 48
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

st.title("Stress Detection")

# File upload section
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Read the uploaded image
    image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the face cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 10)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)

        # Predicting the emotion
        yhat = loaded_model.predict(cropped_img)
        predicted_emotion = labels[int(np.argmax(yhat))]
        cv2.putText(image, predicted_emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        st.image(image, caption=f"Emotion: {predicted_emotion}", use_column_width=True)

