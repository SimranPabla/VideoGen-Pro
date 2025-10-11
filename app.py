from flask import Flask, render_template, request, send_from_directory
import os
import cv2
import numpy as np
import imageio
from werkzeug.utils import secure_filename

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'static', 'outputs')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --Basic Zoom Effect Function--
def zoom_effect(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or unable to read.")
    frames = []    
    h, w, _ = img.shape

    for i in range(10):
        scale = 1 + i * 0.02
        resized = cv2.resize(img, None, fx=scale, fy=scale)
        rh, rw, _ = resized.shape

        #crop back to original size
        top = (rh - h) // 2
        left = (rw - w) // 2
        cropped = resized[top:top+h, left:left+w]

        frames.append(cropped[:, :, ::-1])  # Convert BGR to RGB

    imageio.mimsave(output_path, frames, fps=10)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if not file:
            return "No file uploaded.", 400
        
        filename = secure_filename(file.filename)
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        output_filename = 'animated_' + filename.rsplit('.', 1)[0] + '.gif'
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        file.save(image_path)
        zoom_effect(image_path, output_path)

        return render_template('index.html', result_gif=output_filename)
    
    return render_template('index.html')

@app.route('/outputs/<filename>')
def send_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)