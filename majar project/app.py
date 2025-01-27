import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Initialize Flask app
app = Flask(__name__)

# Load the models and define class labels
models = {
    "plant_disease": {
        "model": tf.keras.models.load_model('plant_disease_model.h5'),
        "class_labels": ['Apple', 'Bell pepper', 'Cherry', 'Citrus', 'Corn', 'Grape', 'Peach', 'Potato', 'Strawberry', 'Tomato']
    },
    "brain_tumor": {
        "model": tf.keras.models.load_model('brain_tumor_model.h5'),
        "class_labels": ['No Tumor', 'Tumor']
    },
    "lung_disease": {
        "model": tf.keras.models.load_model('lung_disease_model.h5'),
        "class_labels": ['Normal', 'Pneumonia', 'COVID-19']
    },
    "breast_cancer": {
        "model": tf.keras.models.load_model('breast_cancer_model.h5'),
        "class_labels": ['Benign', 'Malignant', 'Normal']
    },
    "heart_disease": {
        "model": tf.keras.models.load_model('heart_disease_model.h5'),
        "class_labels": ['Healthy', 'Disease']
    }
}

# Set a confidence threshold
confidence_threshold = 0.6

# Path for saving uploaded images
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def load_and_preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize to [0, 1] range
    return img_array

def predict_image_class(selected_model, image_path, confidence_threshold):
    model = models[selected_model]["model"]
    class_labels = models[selected_model]["class_labels"]
    img_array = load_and_preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)
    confidence = predictions[0][predicted_class_index[0]]

    if confidence < confidence_threshold:
        return "Unknown"
    else:
        return class_labels[predicted_class_index[0]]

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html', models=list(models.keys()))

@app.route('/upload', methods=['GET'])
def upload():
    selected_model = request.args.get('model')
    if selected_model not in models:
        return redirect(url_for('index'))
    return render_template('upload.html', model=selected_model)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or 'model' not in request.form:
        return redirect(request.url)

    file = request.files['file']
    selected_model = request.form['model']

    if file.filename == '' or selected_model not in models:
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        predicted_label = predict_image_class(selected_model, filepath, confidence_threshold)

        return render_template('result.html', predicted_label=predicted_label, model=selected_model)

if __name__ == "__main__":
    app.run(debug=True)
