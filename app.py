from flask import Flask, request, jsonify, send_file
import os
import uuid
import base64
import json
from PIL import Image, ImageFilter
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import pyttsx3

app = Flask(__name__)

# Image Blurring Functionality
@app.route('/blur_image', methods=['POST'])
def blur_image():
    if 'image' not in request.files:
        return jsonify({"status": "failed", "error": "No image file provided"}), 400
    
    image_file = request.files['image']
    try:
        image = Image.open(image_file)
        blurred_image = image.filter(ImageFilter.GaussianBlur(5))
        blurred_image_filename = str(uuid.uuid4()) + ".jpg"
        save_directory = './blurred_images/'
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        blurred_image_path = os.path.join(save_directory, blurred_image_filename)
        blurred_image.save(blurred_image_path)
        return jsonify({"status": "success", "blurred_image_path": blurred_image_path}), 200
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500

# Base64 Image Upload Functionality
@app.route('/upload_image', methods=['POST'])
def upload_image():
    data = request.json
    if 'imageData' not in data:
        return "Error: No image data received", 400
    
    imageData = data['imageData'].replace('data:image/jpeg;base64,', '')
    try:
        imageBytes = base64.b64decode(imageData)
        with open("./image.jpg", "wb") as f:
            f.write(imageBytes)
        return "Image captured successfully!", 200
    except Exception as e:
        return f"Error saving image: {str(e)}", 500

# Geocoding Functionality
@app.route('/get_coordinates', methods=['GET'])
def get_coordinates():
    location_name = request.args.get('location')
    if not location_name:
        return jsonify({"error": "No location provided"}), 400
    
    geolocator = Nominatim(user_agent="my_geocoder")
    try:
        location = geolocator.geocode(location_name, timeout=10)
        if location:
            return jsonify({"latitude": location.latitude, "longitude": location.longitude}), 200
        else:
            return jsonify({"error": "Location not found"}), 404
    except GeocoderTimedOut:
        return jsonify({"error": "Geocoding service timed out"}), 500
    except GeocoderServiceError as e:
        return jsonify({"error": f"Geocoding service error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

# Iris Prediction Functionality
iris = load_iris()
X, y = iris.data, iris.target
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X, y)

@app.route('/predict_species', methods=['POST'])
def predict_species():
    data = request.json
    try:
        sepal_length = float(data['sepal_length'])
        sepal_width = float(data['sepal_width'])
        petal_length = float(data['petal_length'])
        petal_width = float(data['petal_width'])
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]], dtype=float)
        prediction = model.predict(input_data)[0]
        predicted_species = iris.target_names[prediction]
        return jsonify({"predicted_species": predicted_species}), 200
    except (KeyError, ValueError):
        return "Error: Input values must be numeric and include all required fields.", 400
    except Exception as e:
        return f"Error: {str(e)}", 500

# Sending Emails Functionality
@app.route('/send_emails', methods=['POST'])
def send_emails():
    data = request.json
    subject = data['subject']
    body = data['body']
    recipient_list = data['recipients']
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    login = "bookmates.in@gmail.com"
    password = "gpjwunxmxrmoildj"
    results = []

    def send_email(to_email):
        msg = MIMEMultipart()
        msg['From'] = login
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        try:
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(login, password)
            text = msg.as_string()
            server.sendmail(login, to_email, text)
            server.quit()
            return {"recipient": to_email, "status": "success"}
        except Exception as e:
            return {"recipient": to_email, "status": "failed", "error": str(e)}

    for recipient in recipient_list:
        results.append(send_email(recipient))
    
    return jsonify({"results": results}), 200

# Text to Speech Functionality
@app.route('/text_to_speech', methods=['POST'])
def text_to_speech():
    data = request.json
    message = data.get('message')
    if not message:
        return "Error: No message provided.", 400
    
    try:
        engine = pyttsx3.init()
        engine.say(message)
        engine.runAndWait()
        return f"Success: The message '{message}' was spoken.", 200
    except Exception as e:
        return f"Error: {e}", 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
