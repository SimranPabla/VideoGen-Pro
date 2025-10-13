from flask import Flask, render_template, request, Response, stream_with_context, send_from_directory, session, redirect, url_for, jsonify
import os
from werkzeug.utils import secure_filename
from datetime import timedelta
import whisper
import torch
from moviepy.editor import ImageClip, AudioFileClip, CompositeVideoClip, ColorClip, concatenate_videoclips
import shutil
import uuid
import time
from moviepy.video.fx.resize import resize
from threading import Thread
from multiprocessing import cpu_count

# --- Tunables ---
MAX_WORKERS = max(1, cpu_count() - 1)

# --- Flask setup ---
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", str(uuid.uuid4()))
app.permanent_session_lifetime = timedelta(minutes=60)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB

# --- Paths Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
UPLOAD_FOLDER = os.path.join(STATIC_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(STATIC_DIR, 'outputs')
AUDIO_FOLDER = os.path.join(STATIC_DIR, 'audio')
TMP_FOLDER = os.path.join(BASE_DIR, 'tmp_segments')
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, AUDIO_FOLDER, TMP_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# --- Global Resources ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("base", device=device)
progress_store = {}

# --- Aspect Ratios ---
ASPECT_RATIOS = {
    "16:9": (1920, 1080),
    "1:1": (1080, 1080),
    "9:16": (1080, 1920)
}

# --- Video Generation ---
def fast_video_generation(image_paths, audio_path, output_path, aspect_ratio="16:9", use_zoom=False, task_id=None):
    try:
        audio_clip = AudioFileClip(audio_path)
        audio_duration = audio_clip.duration
        num_images = len(image_paths)
        image_duration = audio_duration / num_images

        frame_w, frame_h = ASPECT_RATIOS.get(aspect_ratio, (1920,1080))
        clips = []

        for i, img_path in enumerate(image_paths):
            if task_id and progress_store.get(task_id) == -1:
                return  # Stop if cancelled
            img_clip = ImageClip(img_path)
            img_w, img_h = img_clip.size
            scale = min(frame_w/img_w, frame_h/img_h)
            img_clip = img_clip.resize((int(img_w*scale), int(img_h*scale)))

            background = ColorClip(size=(frame_w, frame_h), color=(255,255,255)).set_duration(image_duration)
            final_clip = CompositeVideoClip([background, img_clip.set_position("center")]).set_duration(image_duration)

            if use_zoom:
                final_clip = final_clip.resize(lambda t: 1.0 + 0.15*(t/image_duration))

            clips.append(final_clip)

            if task_id:
                progress_store[task_id] = int((i+1)/num_images*80)

        video = concatenate_videoclips(clips, method="compose").set_audio(audio_clip)
        video.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac",
                              temp_audiofile="temp-audio.m4a", remove_temp=True, logger=None)

        if task_id and progress_store.get(task_id) != -1:
            progress_store[task_id] = 100
            progress_store[f"result_{task_id}"] = os.path.basename(output_path)
    except Exception as e:
        if task_id:
            progress_store[task_id] = -1
            progress_store[f"error_{task_id}"] = str(e)

# --- Audio Transcription ---
def transcribe_audio(audio_path):
    try:
        result = model.transcribe(audio_path)
        return result.get("text", "")
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"

# --- Flask Routes ---
@app.route('/', methods=['GET','POST'])
def index():
    uploaded_images = session.get('uploaded_images', [])
    uploaded_audio = session.get('uploaded_audio', None)
    task_id = session.get('task_id', None)
    result_video = session.get('result_video', None)

    if request.method == 'POST':
        # Upload images
        if 'images' in request.files:
            files = request.files.getlist('images')
            uploaded_images = session.get('uploaded_images', [])
            for f in files:
                if f and f.filename != '':
                    filename = secure_filename(f.filename)
                    f.save(os.path.join(UPLOAD_FOLDER, filename))
                    uploaded_images.append(filename)  # Append instead of replacing
            session['uploaded_images'] = uploaded_images
            session.pop('result_video', None)
            session.pop('task_id', None)
            return redirect(url_for('index'))

        # Upload audio
        if 'audio' in request.files:
            file = request.files['audio']
            if file.filename != '':
                filename = secure_filename(file.filename)
                audio_path = os.path.join(AUDIO_FOLDER, filename)
                file.save(audio_path)
                session['uploaded_audio'] = filename

                # Generate transcript in background
                def transcribe_and_store():
                    transcript = transcribe_audio(audio_path)
                    session['audio_transcript'] = transcript
                Thread(target=transcribe_and_store).start()

                return redirect(url_for('index'))

        # Generate video
        if 'generate' in request.form and uploaded_images and uploaded_audio:
            video_settings = {
                'aspect_ratio': request.form.get('aspect_ratio','16:9'),
                'use_zoom': 'zoom_effect' in request.form
            }
            session['video_settings'] = video_settings

            ordered_filenames = request.form.getlist('image_order[]')
            image_paths = [os.path.join(UPLOAD_FOLDER,f) for f in (ordered_filenames if ordered_filenames else uploaded_images)]
            audio_path = os.path.join(AUDIO_FOLDER, uploaded_audio)

            output_filename = f"video_{'zoom' if video_settings['use_zoom'] else 'static'}_{uuid.uuid4().hex[:6]}.mp4"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)

            task_id = str(uuid.uuid4())
            session['task_id'] = task_id
            progress_store[task_id] = 0

            Thread(target=fast_video_generation, args=(
                image_paths, audio_path, output_path,
                video_settings['aspect_ratio'], video_settings['use_zoom'], task_id
            )).start()

            return redirect(url_for('index'))

    return render_template(
        'index.html',
        uploaded_images=uploaded_images,
        uploaded_audio=uploaded_audio,
        result_video=result_video,
        task_id=task_id,
        progress_store=progress_store,
        video_settings=session.get('video_settings', {}),
        audio_transcript=session.get('audio_transcript', "Transcript will appear here after processing.")
    )

# --- Static Files ---
@app.route('/uploads/<filename>')
def send_uploaded(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/outputs/<filename>')
def send_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/audio/<filename>')
def send_audio(filename):
    return send_from_directory(AUDIO_FOLDER, filename)

# --- Reset ---
@app.route('/reset')
def reset():
    task_id = session.get('task_id')
    if task_id:
        progress_store[task_id] = -1
        progress_store[f"error_{task_id}"] = "Cancelled by user via reset."
        progress_store.pop(f"result_{task_id}", None)

    for folder in [UPLOAD_FOLDER, AUDIO_FOLDER, OUTPUT_FOLDER, TMP_FOLDER]:
        for f in os.listdir(folder):
            p = os.path.join(folder,f)
            try:
                if os.path.isdir(p): shutil.rmtree(p, ignore_errors=True)
                else: os.remove(p)
            except: pass

    session.clear()
    return redirect(url_for('index'))

# --- Progress ---
@app.route('/progress/<task_id>')
def progress(task_id):
    def event_stream():
        progress = progress_store.get(task_id)
        while progress is None:
            time.sleep(0.1)
            progress = progress_store.get(task_id)
        while progress<100 and progress>=0:
            yield f"data: {progress}\n\n"
            time.sleep(0.5)
            progress = progress_store.get(task_id,100)
        yield f"data: {progress_store.get(task_id,100)}\n\n"
    return Response(stream_with_context(event_stream()), mimetype='text/event-stream')

# --- Complete Task ---
@app.route('/complete_task/<task_id>')
def complete_task(task_id):
    result_filename = progress_store.get(f"result_{task_id}")
    status = progress_store.get(task_id,0)

    if result_filename and status == 100:
        session['result_video'] = result_filename
        payload = {"status": "ok", "filename": result_filename}
    else:
        payload = {"status": "error", "message": progress_store.get(f"error_{task_id}", "Unknown error")}

    session.pop('task_id',None)
    progress_store.pop(task_id,None)
    progress_store.pop(f"result_{task_id}",None)
    progress_store.pop(f"error_{task_id}",None)
    return jsonify(payload)

if __name__=='__main__':
    app.run(debug=True, threaded=True)
