from flask import Flask, render_template, request, send_from_directory, session, redirect, url_for
import os
from werkzeug.utils import secure_filename
from datetime import timedelta
import whisper
from PIL import Image
from moviepy.editor import ImageClip, AudioFileClip, TextClip, CompositeVideoClip, concatenate_videoclips, ColorClip
from moviepy.video.fx.fadein import fadein
from moviepy.video.fx.fadeout import fadeout

# --- Flask setup ---
app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.permanent_session_lifetime = timedelta(minutes=30)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'static', 'outputs')
AUDIO_FOLDER = os.path.join(BASE_DIR, 'static', 'audio')
FONT_FOLDER = os.path.join(BASE_DIR, 'static', 'fonts')

for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, AUDIO_FOLDER, FONT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Path to font
FONT_PATH = os.path.join(FONT_FOLDER, 'BebasNeue-Regular.ttf')  # ensure exists

# --- Load Whisper model ---
model = whisper.load_model("base")

# --- Image preprocessing ---
def preprocess_images(image_paths):
    """Resize images with white padding to make all images same size."""
    images = [Image.open(p) for p in image_paths]

    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)
    padded_paths = []

    for idx, img in enumerate(images):
        new_img = Image.new("RGB", (max_width, max_height), (255, 255, 255))
        paste_x = (max_width - img.width) // 2
        paste_y = (max_height - img.height) // 2
        new_img.paste(img, (paste_x, paste_y))
        padded_path = os.path.join(UPLOAD_FOLDER, f"padded_{idx}.png")
        new_img.save(padded_path)
        padded_paths.append(padded_path)

    return padded_paths

# --- Optimized video generation ---
def generate_video(image_paths, transcription, audio_path, output_path, fps=24):
    audio_clip = AudioFileClip(audio_path)
    audio_duration = audio_clip.duration

    words = transcription.split()
    num_words = len(words)
    num_images = len(image_paths)

    # Duration per image
    img_duration = audio_duration / num_images if num_images > 0 else audio_duration
    words_per_image = max(1, num_words // num_images)  # block of words per image

    clips = []
    word_index = 0

    for img_index, img_path in enumerate(image_paths):
        clip = ImageClip(img_path).set_duration(img_duration)

        # Smooth zoom (10%) centered
        def zoom_func(t):
            ease = 3*(t/img_duration)**2 - 2*(t/img_duration)**3
            return 1 + 0.1 * ease
        clip = clip.resize(zoom_func).set_position("center")

        # Words for this image block
        start = word_index
        end = min(word_index + words_per_image, num_words)
        text = " ".join(words[start:end])
        word_index = end

        # Create TextClip
        txt_clip = TextClip(
            txt=text,
            fontsize=50,
            color="white",
            font=FONT_PATH,
            method='label'
        ).set_duration(img_duration)

        # Create semi-transparent background rectangle
        padding = 20  # pixels around text
        bg_clip = ColorClip(
            size=(txt_clip.w + padding*2, txt_clip.h + padding*2),
            color=(0, 0, 0)
        ).set_opacity(0.5).set_duration(img_duration)

        # Position background and text at center
        bg_clip = bg_clip.set_position("center")
        txt_clip = txt_clip.set_position("center")

        # Fade in/out
        txt_clip = fadein(txt_clip, 0.1)
        txt_clip = fadeout(txt_clip, 0.1)
        bg_clip = fadein(bg_clip, 0.1)
        bg_clip = fadeout(bg_clip, 0.1)

        # Composite image + background + text
        composite = CompositeVideoClip([clip, bg_clip, txt_clip])
        clips.append(composite)

    if not clips:
        clips = [ImageClip(image_paths[0]).set_duration(audio_duration)]

    final_video = concatenate_videoclips(clips, method="compose").set_audio(audio_clip)
    final_video.write_videofile(output_path, fps=fps, codec="libx264", audio_codec="aac")

# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    uploaded_images = session.get('uploaded_images', [])
    uploaded_audio = session.get('uploaded_audio', None)
    transcription = session.get('transcription', '')

    if request.method == 'POST':
        # Upload images
        if 'images' in request.files:
            files = request.files.getlist('images')
            new_files = []
            for file in files:
                if file and file.filename != '':
                    filename = secure_filename(file.filename)
                    file.save(os.path.join(UPLOAD_FOLDER, filename))
                    new_files.append(filename)
            session['uploaded_images'] = new_files
            return redirect(url_for('index'))

        # Upload audio + transcribe
        if 'audio' in request.files:
            file = request.files['audio']
            if file and file.filename != '':
                filename = secure_filename(file.filename)
                audio_path = os.path.join(AUDIO_FOLDER, filename)
                file.save(audio_path)

                result = model.transcribe(audio_path)
                transcription = result.get('text', '')

                session['uploaded_audio'] = filename
                session['transcription'] = transcription

            return redirect(url_for('index'))

        # Generate video
        if request.form.get('effect') == 'zoom' and uploaded_images and transcription:
            image_paths = [os.path.join(UPLOAD_FOLDER, img) for img in uploaded_images]
            image_paths = preprocess_images(image_paths)
            audio_path = os.path.join(AUDIO_FOLDER, uploaded_audio)
            output_filename = "zoom_video.mp4"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)

            generate_video(image_paths, transcription, audio_path, output_path)

            return render_template(
                'index.html',
                uploaded_images=uploaded_images,
                uploaded_audio=uploaded_audio,
                transcription=transcription,
                result_video=output_filename
            )

    return render_template(
        'index.html',
        uploaded_images=uploaded_images,
        uploaded_audio=uploaded_audio,
        transcription=transcription,
        result_video=None
    )

# --- Serve files ---
@app.route('/uploads/<filename>')
def send_uploaded(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/outputs/<filename>')
def send_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/audio/<filename>')
def send_audio(filename):
    return send_from_directory(AUDIO_FOLDER, filename)

@app.route('/reset')
def reset():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
