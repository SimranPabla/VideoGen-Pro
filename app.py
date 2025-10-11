from flask import Flask, render_template, request, send_from_directory
import os
import cv2
import numpy as np
import imageio

app = Flask(__name__)

UPLOAD_FOLDER = '/static/uploads/'
OUTPUT_FOLDER = '/static/outputs/'
os.makedirs('static/uploads', exist_ok=True)
os.makedirs('static/outputs', exist_ok=True)

# --Basic Zoom Effect Function--
def zoom_effect(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or unable to read.")
    frames = []    
    h, w, _ = img.shape

    for i in range(10):
        scale = 1 + i * 0.2
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
        
        filename = file.filename
        image_path = os.path.join('static/UPLOAD_FOLDER', filename)
        output_filename = 'animated_' + filename.rsplit('.', 1)[0] + '.gif'
        output_path = os.path.join('static/OUTPUT_FOLDER', output_filename)
        
        file.save(image_path)
        zoom_effect(image_path, output_path)

        return render_template('index.html', output_image=output_filename)
    
    return render_template('index.html')

@app.route('/static/<filename>')
def send_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)