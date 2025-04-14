from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

file_path = "plant_disease_model.keras"
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
else:
    print(f"File found: {file_path}")

app = Flask(__name__)

# Load the trained model
#model = load_model("plant_disease_model.keras")


# Class labels (replace these with your actual class names)
#class_labels = ['Class 1', 'Class 2']  # Update these to your class names
class_labels = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']

"""
class_labels = [
    'Powdery Mildew',  # Class 0
    'Rust Disease',    # Class 1
    'Early Blight',    # Class 2
    'Late Blight',     # Class 3
    'Healthy Plant'    # Class 4
]
"""

def preprocess_image(image_path):
    # Preprocess the image to match the input format
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0  # Rescale as done during training
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            # Save the uploaded image to a static folder
            filepath = os.path.join("static", file.filename)
            file.save(filepath)
            
            # Preprocess the image and make a prediction
            img_array = preprocess_image(filepath)
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = predictions[0][predicted_class]
            
            result = {
                "class_label": class_labels[predicted_class],
                "confidence": f"{confidence * 100:.2f}%",
                "image_path": filepath
            }
            
            return render_template("index.html", result=result)
    
    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)
