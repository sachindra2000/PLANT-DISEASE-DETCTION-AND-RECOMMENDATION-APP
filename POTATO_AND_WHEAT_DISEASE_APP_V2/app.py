
import logging
import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array,ImageDataGenerator
from PIL import Image

# Create the Flask app
app = Flask(__name__)

# Define class labels
wheat_class_labels = ['Aphid', 'Black Rust', 'Blast', 'Brown Rust', 'Common Root Rot', 'Fusarium Head Blight', 'Healthy', 'Leaf Blight', 'Mildew', 'Mite', 'Septoria', 'Smut', 'Stem Fly', 'Tan Spot', 'Yellow Rust']
potato_class_labels = ['Bacteria', 'Fungi', 'Healthy', 'Nematode', 'Pest', 'Phytophthora', 'Virus']

# Load models with error handling
potato_model_path = "potato_disease_model test accuracy 0.7652733325958252.keras"
wheat_model_path = "wheat_disease_model accuracy 0.8933.keras"

potato_model = None
wheat_model = None

if os.path.exists(potato_model_path):
    try:
        potato_model = load_model(potato_model_path)
        logging.info("Potato model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading potato model: {e}")

if os.path.exists(wheat_model_path):
    try:
        wheat_model = load_model(wheat_model_path)
        logging.info("Wheat model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading wheat model: {e}")

# Exit if models failed to load
if potato_model is None or wheat_model is None:
    logging.error("One or more models failed to load. Exiting...")
    exit(1)

# Preprocessing function
def preprocess_image(file, target_size=(255, 255)):
    try:
        #img = Image.open(file.stream).convert("RGB")
        img = Image.open(file)
        img = img.resize(target_size)
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)  # Expand dimensions to match model input
        img = img / 255.0  # Normalize

        #datagen = ImageDataGenerator(rescale=1./255)
        #img = datagen.standardize(img)
        return img
    except Exception as e:
        raise ValueError(f"Image processing failed: {e}")
    
# Add recommendations for each class
potato_recommendations = {
    'Bacteria': 'Use bactericides and ensure proper sanitation.',
    'Fungi': 'Apply fungicides and maintain proper soil drainage.',
    'Healthy': 'No action needed. Your crop is healthy.',
    'Nematode': 'Use nematicides and practice crop rotation.',
    'Pest': 'Apply appropriate pesticides and monitor crop regularly.',
    'Phytophthora': 'Improve soil drainage and use resistant varieties.',
    'Virus': 'Remove infected plants and control insect vectors.'
}

wheat_recommendations = {
    'Aphid': 'Use insecticides and encourage natural predators.',
    'Black Rust': 'Apply fungicides and use resistant varieties.',
    'Blast': 'Use fungicides and ensure proper field sanitation.',
    'Brown Rust': 'Apply fungicides and use resistant varieties.',
    'Common Root Rot': 'Improve soil drainage and use crop rotation.',
    'Fusarium Head Blight': 'Apply fungicides and use resistant varieties.',
    'Healthy': 'No action needed. Your crop is healthy.',
    'Leaf Blight': 'Apply fungicides and ensure proper field sanitation.',
    'Mildew': 'Use fungicides and ensure proper air circulation.',
    'Mite': 'Use miticides and encourage natural predators.',
    'Septoria': 'Apply fungicides and use resistant varieties.',
    'Smut': 'Use fungicides and ensure proper seed treatment.',
    'Stem Fly': 'Use insecticides and monitor crop regularly.',
    'Tan Spot': 'Apply fungicides and use resistant varieties.',
    'Yellow Rust': 'Apply fungicides and use resistant varieties.'
}
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    file = request.files["file"]
    if file.filename == "" or not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        return jsonify({"error": "Invalid file type. Please upload a PNG or JPG image."})
    
    model_type = request.form.get("model_type", "potato")
    
    try:
        img = preprocess_image(file)
        logging.debug(f"Image preprocessed: {img.shape}")
    except ValueError as e:
        return jsonify({"error": str(e)})
    
    if model_type == "potato":
        model = potato_model
        class_labels = potato_class_labels
        recommendations = potato_recommendations
    elif model_type == "wheat":
        model = wheat_model
        class_labels = wheat_class_labels
        recommendations = wheat_recommendations
    else:
        return jsonify({"error": "Invalid model type"})
    
    try:
        prediction = model.predict(img)
        logging.debug(f"Class probabilities: {prediction.tolist()}")
        predicted_class_index = int(np.argmax(prediction, axis=1)[0])
        confidence = float(np.max(prediction))
        predicted_class_name = class_labels[predicted_class_index]
        recommendation = recommendations[predicted_class_name]

        response_data = {
            "predicted_class": predicted_class_name,
            "confidence": confidence,
            "recommendations": recommendation
        }
        logging.debug(f"Response JSON: {response_data}")  # Log final response
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"})
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)