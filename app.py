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
def apply_effect(image_path, output_path, effect):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found or unable to read.")
    frames = []    
    h, w, _ = img.shape
    num_frames = 10

    for i in range(num_frames):
        if effect == "zoom":
            scale = 1 + i * 0.02
            resized = cv2.resize(img, None, fx=scale, fy=scale)
            rh, rw, _ = resized.shape

            #crop back to original size
            top = (rh - h) // 2
            left = (rw - w) // 2
            frame = resized[top:top+h, left:left+w]
        
        elif effect == "rotate":
            angle = i * (360 / (num_frames - 1))
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
            frame = cv2.warpAffine(img, M, (w, h))

        elif effect == "pan":
            shift = int(i * (w / (num_frames - 1)))
            frame = np.roll(img, shift, axis=1)  # shift horizontally

  
        elif effect == "fade":
            alpha = i / (num_frames - 1)
            frame = cv2.convertScaleAbs(img, alpha=alpha)

        else:
            frame = img

        frames.append(frame[:, :, ::-1])  # Convert BGR to RGB

    imageio.mimsave(output_path, frames, fps=10)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if not file:
            return "No file uploaded.", 400
        
        effect = request.form.get('effect', 'zoom')
        filename = secure_filename(file.filename)
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        output_filename = f"{effect}animated_" + filename.rsplit('.', 1)[0] + ".gif"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        file.save(image_path)
        apply_effect(image_path, output_path,effect)

        return render_template('index.html', result_gif=output_filename)
    
    return render_template('index.html')

@app.route('/outputs/<filename>')
def send_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)