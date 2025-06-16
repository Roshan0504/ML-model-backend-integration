
from flask import Flask, render_template, request, redirect, url_for
import os
print("TEMPLATES FOUND:", os.listdir('templates'))
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import numpy as np

import tensorflow as tf

from google.cloud import storage

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load your ML model
# model  = load_model('rice.h5')


def download_model_from_gcs(bucket_name, blob_name, destination_file_name):
    if not os.path.exists(destination_file_name):
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(destination_file_name)
        print(f"Downloaded model from GCS: {destination_file_name}")
    else:
        print("Model already exists locally.")


download_model_from_gcs('rice_bucket_model','rice.h5','rice.h5'   )
model = tf.keras.models.load_model('rice.h5')
#gcs_model_path = 'gs://rice_bucket_model/rice.h5' 
#model = tf.keras.models.load_model(gcs_model_path)

CLASS_NAMES = ['Bacterial leaf blight','Brown Spot','Leaf smut', 'Healthy']  # Update with your classes

@app.route('/')
def start_page():
    return render_template('index.html')

@app.route('/upload')
def upload_page():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict_page():
    if 'file' not in request.files:
        return redirect(url_for('upload_page'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('upload_page'))
    
    if file:
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        print("After to_array:", img_array.shape)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        print("Final input shape for model:", img_array.shape)
        
        pred = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(pred[0])]
        confidence = round(float(np.max(pred[0])) * 100, 2)
        
        return render_template('predict.html', 
                            image_path=url_for('static', filename='uploads/' + filename),
                           prediction=predicted_class,
                           confidence=confidence)
    
    return redirect(url_for('upload_page'))


    
