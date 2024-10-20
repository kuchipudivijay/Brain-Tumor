
from flask import Flask, render_template, request
import cv2
import os
import numpy as np
import pickle
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))  # Resize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize pixel values
    return image

# Function to perform prediction
def predict(image):
    try:
        import tensorflow.keras.backend as K
        # Clear TensorFlow session
        K.clear_session()

        preprocessed_img = preprocess_image(image)
        model = load_model("brain_tumor_detection_model.h5")  # Load the model inside the function
        with open("label_binarizer.pkl", "rb") as f:
            label_binarizer = pickle.load(f)

        prediction = model.predict(preprocessed_img)
        class_idx = np.argmax(prediction)
        label = label_binarizer.classes_[class_idx]
        return label
    except Exception as e:
        return "Error"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return render_template('index.html', prediction='No file found')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', prediction='No file selected')

    try:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        label = predict(img)
        return render_template('index.html', prediction=f'Tumor type: {label}')
    except Exception as e:
        return render_template('index.html', prediction=f'Error occurred: {e}')

if __name__ == '__main__':
    app.run(debug=True)
