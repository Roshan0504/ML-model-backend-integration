from flask import Flask, render_template, request, redirect, url_for
import os
import logging
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from google.cloud import storage

# Logging setup
logging.basicConfig(level=logging.INFO)

# Initialize Flask app
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# GCS Storage
storage_client = storage.Client()

# Model download path
MODEL_LOCAL_PATH = '/tmp/rice.h5'

def download_model_from_gcs(bucket_name, blob_name, destination_file_name):
    try:
        if not os.path.exists(destination_file_name):
            logging.info("Downloading model from GCS...")
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(destination_file_name)
            logging.info(f"Downloaded model to {destination_file_name}")
        else:
            logging.info("Model already exists locally.")
    except Exception as e:
        logging.error(f"Error downloading model from GCS: {e}", exc_info=True)

# Download and load model
download_model_from_gcs('rice_bucket_model', 'rice.h5', MODEL_LOCAL_PATH)
try:
    model = load_model(MODEL_LOCAL_PATH)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}", exc_info=True)

# Class labels
CLASS_NAMES = ['Bacterial leaf blight', 'Brown Spot', 'Leaf smut', 'Healthy']

@app.route('/')
def start_page():
    logging.info("Rendering start page.")
    return render_template('index.html')

@app.route('/upload')
def upload_page():
    logging.info("Rendering upload page.")
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict_page():
    try:
        if 'file' not in request.files:
            logging.warning("No file in request.")
            return redirect(url_for('upload_page'))

        file = request.files['file']
        if file.filename == '':
            logging.warning("Empty filename.")
            return redirect(url_for('upload_page'))

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            logging.info(f"File saved to {filepath}")

            # Preprocess image
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            logging.info("Making prediction...")
            pred = model.predict(img_array)
            predicted_class = CLASS_NAMES[np.argmax(pred[0])]
            confidence = round(float(np.max(pred[0])) * 100, 2)

            return render_template('predict.html',
                                   image_path=url_for('static', filename='uploads/' + filename),
                                   prediction=predicted_class,
                                   confidence=confidence)
    except Exception as e:
        logging.error(f"Prediction failed: {e}", exc_info=True)
        return "Internal Server Error", 500

    return redirect(url_for('upload_page'))
