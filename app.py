from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
import cv2
import base64
import re
import os

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model('my_digit_model.h5')

def preprocess_and_segment(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold to binary (invert for white on black)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Morphological close to help with noisy images
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours - external only
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digit_imgs = []
    bounding_boxes = []

    # Filter contours by size and extract bounding boxes
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 5 and h > 5:  # filter tiny noise
            bounding_boxes.append((x, y, w, h))

    if not bounding_boxes:
        return []

    # Determine layout by analyzing bounding box distribution
    xs = [box[0] for box in bounding_boxes]
    ys = [box[1] for box in bounding_boxes]
    x_range = max(xs) - min(xs)
    y_range = max(ys) - min(ys)

    # Sort bounding boxes by x (left to right) if horizontal
    # or by y (top to bottom) if vertical based on range
    if x_range >= y_range:
        bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])  # horizontal
    else:
        bounding_boxes = sorted(bounding_boxes, key=lambda b: b[1])  # vertical

    # Extract each digit image, resize to 28x28
    for box in bounding_boxes:
        x, y, w, h = box
        digit_img = gray[y:y+h, x:x+w]
        resized_digit = cv2.resize(digit_img, (28, 28), interpolation=cv2.INTER_AREA)
        digit_imgs.append(resized_digit)

    return digit_imgs

def prepare_image_for_model(digit_img):
    # Normalize and reshape to (1, 28, 28, 1)
    norm_img = digit_img.astype('float32') / 255.0
    norm_img = norm_img.reshape(1, 28, 28, 1)
    return norm_img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if file upload
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Invalid image'}), 400
        
    else:
        # Check if drawing data submitted
        data = request.json.get('image')
        if not data:
            return jsonify({'error': 'No image data provided'}), 400
        
        img_str = re.search(r'base64,(.*)', data).group(1)
        img_bytes = base64.b64decode(img_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Invalid image data'}), 400

    # Segment digits from the image
    digit_imgs = preprocess_and_segment(img)
    if not digit_imgs:
        return jsonify({'error': 'No digits found'}), 400

    predicted_digits = []
    for digit_img in digit_imgs:
        input_img = prepare_image_for_model(digit_img)
        prediction = model.predict(input_img)
        digit = int(np.argmax(prediction))
        predicted_digits.append(str(digit))

    # Join predicted digits (horizontal or vertical sequence)
    predicted_number = ''.join(predicted_digits)

    return jsonify({'predicted_number': predicted_number})
if __name__ == '__main__':
    # Run on localhost port 5000 with debug
    app.run(debug=True)
