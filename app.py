from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import load_model
import json
import os

app = Flask(__name__)

# Update this path to correctly reference the static/uploads directory
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static', 'uploads')
app.secret_key = 'supersecretkey'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load class indices
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
class_names = {v: k for k, v in class_indices.items()}

# Load VGG16 model pre-trained on ImageNet without the top classifier
vgg16 = VGG16(include_top=False, weights='imagenet')

# Load the custom model for classification
model = load_model(r'C:\Users\pravi\Desktop\major project\flask application\arecanut disease identification\model.h5')  # Replace 'model.h5' with the actual path to your custom model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Load and preprocess the image
        test_image = load_img(file_path, target_size=(224, 224))
        test_image = img_to_array(test_image)
        test_image = test_image / 255.0
        test_image = np.expand_dims(test_image, axis=0)
        
        # Predict using VGG16
        bt_prediction = vgg16.predict(test_image)
        
        # Predict using custom model
        preds = model.predict(bt_prediction)
        predicted_class_index = np.argmax(preds)
        
        # Get the predicted class name and confidence
        predicted_class_name = class_names[predicted_class_index]
        confidence = preds[0][predicted_class_index]
        
        if confidence < 0.8:
            result = "It does not seem like an areca nut"
            detailed_link = None
        else:
            result = f"Predicted as {predicted_class_name}"
            detailed_link = url_for('disease_detail', disease=predicted_class_name.replace(' ', '_'))
        
        image_url = url_for('static', filename=f'uploads/{filename}')
        return render_template('results.html', result=result, image_url=image_url, detailed_link=detailed_link)
    return redirect(url_for('home'))

@app.route('/disease/<disease>')
def disease_detail(disease):
    return render_template(f'{disease}.html')

if __name__ == '__main__':
    app.run(debug=True)
