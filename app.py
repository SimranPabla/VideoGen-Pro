from flask import Flask, render_template, request, send_from_directory, session, redirect, url_for, jsonify
import os
import cv2
import numpy as np
import imageio
from werkzeug.utils import secure_filename
from datetime import timedelta

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.permanent_session_lifetime = timedelta(minutes=30)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'static', 'outputs')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- EFFECT FUNCTION ---
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
            top = (rh - h) // 2
            left = (rw - w) // 2
            frame = resized[top:top + h, left:left + w]

        elif effect == "rotate":
            angle = i * (360 / (num_frames - 1))
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
            frame = cv2.warpAffine(img, M, (w, h))

        elif effect == "pan":
            shift = int(i * (w / (num_frames - 1)))
            frame = np.roll(img, shift, axis=1)

        elif effect == "fade":
            alpha = i / (num_frames - 1)
            frame = cv2.convertScaleAbs(img, alpha=alpha)

        else:
            frame = img

        frames.append(frame[:, :, ::-1])  # Convert BGR → RGB

    imageio.mimsave(output_path, frames, fps=10)


# --- ROUTES ---
@app.route('/', methods=['GET', 'POST'])
def index():
    uploaded_image = session.get('uploaded_image')

    if request.method == 'POST':
        # File upload
        if 'image' in request.files and request.files['image'].filename != '':
            file = request.files['image']
            filename = secure_filename(file.filename)
            image_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(image_path)
            session.permanent = True
            session['uploaded_image'] = filename
            return redirect(url_for('index') + "#result-section")

        # Apply effect
        effect = request.form.get('effect')
        if effect and uploaded_image:
            image_path = os.path.join(UPLOAD_FOLDER, uploaded_image)
            output_filename = f"{effect}_animated_" + uploaded_image.rsplit('.', 1)[0] + ".gif"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)

            apply_effect(image_path, output_path, effect)
            return render_template('index.html',
                                   uploaded_image=uploaded_image,
                                   result_gif=output_filename,
                                   scroll_to="result-section")

    return render_template('index.html', uploaded_image=uploaded_image, result_gif=None)


@app.route('/uploads/<filename>')
def send_uploaded(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/outputs/<filename>')
def send_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


@app.route('/reset')
def reset():
    session.pop('uploaded_image', None)
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
