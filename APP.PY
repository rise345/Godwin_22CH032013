from flask import Flask, render_template, request, jsonify, url_for
import numpy as np
import cv2
import tensorflow as tf
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_FILE = "face_emotionModel_compat.tflite"

# Load TFLite interpreter
interpreter = tf.lite.Interpreter(model_path=MODEL_FILE)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Default emotion labels (7-class). If your model uses a different order, update this list.
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def preprocess_image(img_bgr):
    # Convert to grayscale, resize to 48x48, normalize to 0-1 float32
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized.astype(np.float32) / 255.0
    # Many TFLite models expect shape (1,48,48,1)
    img = np.expand_dims(normalized, axis=0)
    if len(input_details[0]['shape']) == 4:
        img = np.expand_dims(normalized, axis=-1)
    return img

def predict_emotion_from_image(img_bgr):
    img = preprocess_image(img_bgr)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    pred = int(np.argmax(output_data))
    label = EMOTIONS[pred] if pred < len(EMOTIONS) else str(pred)
    return label

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    # Read image with OpenCV
    file_bytes = np.frombuffer(open(save_path,'rb').read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({'error': 'Could not read image'}), 400

    emotion = predict_emotion_from_image(image)
    return jsonify({'emotion': emotion, 'image_url': url_for('static', filename='uploads/' + filename)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
