from flask import Flask, render_template, request, Response, stream_with_context, send_from_directory, session, redirect, url_for
import os
from werkzeug.utils import secure_filename
from datetime import timedelta
import whisper
import torch
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageClip, AudioFileClip, CompositeVideoClip
import subprocess
import tempfile
import shutil
import uuid
import numpy as np
import time
import sys
from threading import Thread
import math
from multiprocessing import Pool, cpu_count
import traceback

# --- Tunables ---
SEGMENT_MIN_DURATION = 0.5
MAX_WORKERS = max(1, cpu_count() - 1)

# --- Flask setup ---
app = Flask(__name__)
# Load config from environment or default to secure values
app.secret_key = os.environ.get("FLASK_SECRET_KEY", str(uuid.uuid4()))
app.permanent_session_lifetime = timedelta(minutes=60)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024 # 200 MB uploads

# --- Paths Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
UPLOAD_FOLDER = os.path.join(STATIC_DIR, 'uploads')
OUTPUT_FOLDER = os.path.join(STATIC_DIR, 'outputs')
AUDIO_FOLDER = os.path.join(STATIC_DIR, 'audio')
FONT_FOLDER = os.path.join(STATIC_DIR, 'fonts')
TMP_FOLDER = os.path.join(BASE_DIR, 'tmp_segments')

for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, AUDIO_FOLDER, FONT_FOLDER, TMP_FOLDER]:
    os.makedirs(folder, exist_ok=True)

FONT_PATH = os.path.join(FONT_FOLDER, 'BebasNeue-Regular.ttf')

# --- Global Resources ---
FFMPEG_CODEC = "libx264" # Enforcing CPU software encoding
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[startup] Whisper device: {device}", file=sys.stderr)
model = whisper.load_model("base", device=device)
print("[startup] Whisper model loaded", file=sys.stderr)

progress_store = {} # Tracks video generation progress per task

# --- Aspect Ratios ---
ASPECT_RATIOS = {
    "16:9": (1920, 1080),
    "1:1": (1080, 1080),
    "9:16": (1080, 1920)
}

# --- Core Video Functions (Updated for Zoom/Static) ---

def create_static_overlay_clip(segment_text, font_size=50, text_position="bottom", canvas_size=(1920,1080), max_width=1720, duration=1.0):
    # (Implementation remains the same as it's complex and functional)
    W, H = canvas_size
    padding = 20
    box_radius = 16
    box_alpha = 160
    text_color = (255, 255, 255, 255)
    box_color = (0, 0, 0, box_alpha)

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except Exception:
        font = ImageFont.load_default()

    lines = []
    current = ""
    words = segment_text.split()

    for w in words:
        test = w if current == "" else current + " " + w
        bbox = draw.textbbox((0, 0), test, font=font)
        if (bbox[2] - bbox[0]) <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)

    line_heights = []
    line_spacing = int(font_size * 0.25)

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_heights.append(bbox[3] - bbox[1])
    
    line_height = max(line_heights) if line_heights else font_size
    text_block_w = max(draw.textbbox((0, 0), line, font=font)[2] - draw.textbbox((0, 0), line, font=font)[0] for line in lines) if lines else 0
    text_block_h = len(lines) * line_height + (len(lines) - 1) * line_spacing

    box_w = text_block_w + 2*padding
    box_h = text_block_h + 2*padding

    if text_position == "top":
        box_y = 50
    elif text_position == "center":
        box_y = (H - box_h) // 2
    else:
        box_y = H - box_h - 50
    box_x = (W - box_w) // 2

    try:
        draw.rounded_rectangle([box_x, box_y, box_x+box_w, box_y+box_h], radius=box_radius, fill=box_color)
    except Exception:
        draw.rectangle([box_x, box_y, box_x+box_w, box_y+box_h], fill=box_color)

    ty = box_y + padding
    for line in lines:
        line_w = draw.textbbox((0,0), line, font=font)[2] - draw.textbbox((0,0), line, font=font)[0]
        tx = box_x + padding + (box_w - 2 * padding - line_w) // 2
        draw.text((tx, ty), line, font=font, fill=text_color)
        ty += line_height + line_spacing

    overlay_array = np.array(overlay, dtype=np.uint8)
    clip = ImageClip(overlay_array).set_duration(duration)
    return clip

def create_image_base_clip(img_path, duration, aspect_ratio="16:9", zoom_start=1.0, zoom_end=1.15, use_zoom=True):
    """
    Creates the base ImageClip, applying padding and optional zoom effect.
    This replaces the original create_zoom_clip.
    """
    frame_w, frame_h = ASPECT_RATIOS.get(aspect_ratio, (1920, 1080))
    
    img = Image.open(img_path).convert("RGB")
    img_w, img_h = img.size
    target_ratio = frame_w / frame_h
    img_ratio = img_w / img_h

    # Calculate padding/resizing to fit within the aspect ratio
    if img_ratio > target_ratio:
        new_w = frame_w
        new_h = int(frame_w / img_ratio)
    else:
        new_h = frame_h
        new_w = int(frame_h * img_ratio)

    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Apply padding/background
    bg = Image.new("RGB", (frame_w, frame_h), (255, 255, 255))
    x_offset = (frame_w - new_w) // 2
    y_offset = (frame_h - new_h) // 2
    bg.paste(img_resized, (x_offset, y_offset))
    
    padded_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.png")
    bg.save(padded_path)

    clip = ImageClip(padded_path).set_duration(duration)
    
    if use_zoom:
        clip = clip.resize(lambda t: zoom_start + (zoom_end - zoom_start) * (t / duration))
    
    clip = clip.set_position('center')
    os.remove(padded_path) 
    return clip

def create_segment_file_wrapper(args):
    """Wrapper for multiprocessing Pool."""
    # Added 'use_zoom' to the arguments tuple
    index, img_path, segment_words, segment_times, font_size, text_position, tmp_dir, fps, aspect_ratio, use_zoom = args
    return create_segment_file(index, img_path, segment_words, segment_times, font_size, text_position, tmp_dir, fps, aspect_ratio, use_zoom)

def create_segment_file(index, img_path, segment_words, segment_times, font_size, text_position, tmp_dir, fps, aspect_ratio="16:9", use_zoom=True):
    """Create a single video segment."""
    seg_filename = os.path.join(tmp_dir, f"segment_{index:06d}.mp4")
    seg_duration = segment_times[-1][1] 
    segment_text = " ".join(segment_words)
    frame_w, frame_h = ASPECT_RATIOS.get(aspect_ratio, (1920, 1080))

    # Determine if we use the zoom effect or a static image
    base_clip = create_image_base_clip(img_path, seg_duration, aspect_ratio, use_zoom=use_zoom)
    static_overlay_clip = create_static_overlay_clip(segment_text, font_size=font_size, text_position=text_position, canvas_size=(frame_w, frame_h), duration=seg_duration).set_position("center")

    segment_clip = CompositeVideoClip([base_clip, static_overlay_clip], size=(base_clip.w, base_clip.h)).set_duration(seg_duration)

    segment_clip.write_videofile(
        seg_filename, fps=fps, codec=FFMPEG_CODEC, audio=False, threads=1, preset="medium", logger=None, ffmpeg_params=["-pix_fmt", "yuv420p"]
    )

    for clip in [segment_clip, base_clip, static_overlay_clip]:
        try:
            clip.close()
        except:
            pass

    return seg_filename

def _ffmpeg_concat_entry(path):
    return f"file '{path.replace("'", "'\\''")}'\n"

def concat_segments_copy_video(segment_paths, audio_path, output_path, task_id=None):
    if task_id: progress_store[task_id] = 85
    if not segment_paths: raise ValueError("No segments to concatenate")

    tmp_dir = os.path.dirname(segment_paths[0])
    list_file = os.path.join(tmp_dir, f"concat_{uuid.uuid4().hex}.txt")
    with open(list_file, "w", encoding="utf-8") as f:
        for p in segment_paths:
            f.write(_ffmpeg_concat_entry(p))

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-f", "concat", "-safe", "0", "-i", list_file,
        "-i", audio_path, "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", output_path
    ]
    subprocess.run(cmd, check=True)
    os.remove(list_file)
    if task_id: progress_store[task_id] = 95

# --- Main Background Task Runner (Updated signature) ---

def video_generation_task(
        image_paths, transcription, audio_path, output_path,
        font_size, text_position, fps, segments,
        aspect_ratio, words_per_chunk, task_id, output_filename, use_zoom): # Added use_zoom
    
    try:
        audio_clip = AudioFileClip(audio_path)
        audio_duration = audio_clip.duration
        audio_clip.close()

        # --- Split transcription into segments ---
        caption_segments = []
        if segments and len(segments) > 0:
            for seg in segments:
                start = float(seg.get("start", 0.0))
                end = float(seg.get("end", start + 1.0))
                text = seg.get("text", "").strip()
                segment_words = text.split() 
                if not segment_words: continue

                word_details = seg.get('words', [])
                if word_details:
                    # Segment duration for clip (relative)
                    word_times = [(0, end - start)]
                else:
                    word_times = [(0, end - start)]

                caption_segments.append({"words": segment_words, "word_times": word_times})
        else:
            # Fallback for missing Whisper segments
            words = transcription.split()
            chunk_size = words_per_chunk 
            num_chunks = math.ceil(len(words)/chunk_size)
            chunk_dur = max(audio_duration / max(1,num_chunks), SEGMENT_MIN_DURATION)
            for i in range(0, len(words), chunk_size):
                caption_segments.append({"words": words[i:i+chunk_size], "word_times": [(0, chunk_dur)]})

        if not caption_segments:
            raise ValueError("No caption segments to render")

        image_for_segment = [image_paths[i % len(image_paths)] for i in range(len(caption_segments))]
        tmp_dir = tempfile.mkdtemp(dir=TMP_FOLDER)
        total_segments = len(caption_segments) 
        pool_args = []

        for i, seg in enumerate(caption_segments):
            # Added use_zoom here:
            pool_args.append((i, image_for_segment[i], seg["words"], seg["word_times"], font_size, text_position, tmp_dir, fps, aspect_ratio, use_zoom))

        # --- Segment generation (Parallel) ---
        MAX_SEGMENT_PROGRESS = 80
        segment_paths = []
        with Pool(processes=MAX_WORKERS) as pool:
            for i, seg_file in enumerate(pool.imap_unordered(create_segment_file_wrapper, pool_args)):
                segment_paths.append(seg_file)
                if task_id:
                    progress_store[task_id] = int(((i+1)/total_segments) * MAX_SEGMENT_PROGRESS)

        segment_paths.sort() 

        # --- Concat segments with audio ---
        concat_segments_copy_video(segment_paths, audio_path, output_path, task_id=task_id)

        # --- Cleanup ---
        shutil.rmtree(tmp_dir, ignore_errors=True)
        progress_store[task_id] = 100 
        progress_store[f"result_{task_id}"] = output_filename
        
    except Exception as e:
        error_info = traceback.format_exc()
        print(f"[FATAL ERROR] Video generation failed for task {task_id}: {e}\n{error_info}", file=sys.stderr)
        progress_store[task_id] = -1 
        progress_store[f"error_{task_id}"] = str(e)


# --- Flask routes (Updated index route) ---

@app.route('/', methods=['GET', 'POST'])
def index():
    uploaded_images = session.get('uploaded_images', [])
    uploaded_audio = session.get('uploaded_audio', None)
    transcription = session.get('transcription', '')
    segments = session.get('segments', [])
    task_id = session.get('task_id', None)
    result_video = session.get('result_video', None)

    if request.method == 'POST':
        # 1. Upload Images
        if 'images' in request.files:
            files = request.files.getlist('images')
            new_files = [secure_filename(f.filename) for f in files if f and f.filename != '']
            for file in files:
                if file and file.filename != '':
                    file.save(os.path.join(UPLOAD_FOLDER, secure_filename(file.filename)))
            session['uploaded_images'] = new_files
            session.pop('result_video', None) 
            session.pop('task_id', None)
            return redirect(url_for('index'))

        # 2. Upload Audio and Transcribe
        if 'audio' in request.files:
            file = request.files['audio']
            if file and file.filename != '':
                filename = secure_filename(file.filename)
                audio_path = os.path.join(AUDIO_FOLDER, filename)
                file.save(audio_path)

                trans_task_id = str(uuid.uuid4())
                session['task_id'] = trans_task_id 
                progress_store[trans_task_id] = 5
                
                try:
                    result = model.transcribe(audio_path, verbose=False)
                    progress_store[trans_task_id] = 100 

                    session['uploaded_audio'] = filename
                    session['transcription'] = result.get('text', '')
                    session['segments'] = result.get('segments', [])

                except Exception as e:
                    print(f"[ERROR] Transcription failed: {e}", file=sys.stderr)
                    progress_store[trans_task_id] = -1 
                    session['uploaded_audio'] = None
                    session['transcription'] = "Error: Transcription failed."

                session.pop('result_video', None) 
                session.pop('task_id', None) 
                return redirect(url_for('index'))

        # 3. Generate Video (Checking for 'generate' which is the new button name)
        if 'generate' in request.form and uploaded_images and uploaded_audio and transcription:
            
            # Prevent re-submitting if a task is already running/pending completion
            if task_id and progress_store.get(task_id, 100) < 100:
                return redirect(url_for('index'))

            # --- NEW LOGIC: Check for zoom effect ---
            use_zoom = 'zoom_effect' in request.form
            
            font_size = int(request.form.get('font_size', 50))
            text_position = request.form.get('text_position', 'bottom')
            user_aspect_ratio = request.form.get("aspect_ratio", "16:9") 
            words_per_chunk = int(request.form.get('words_per_chunk', 5)) 

            image_paths = [os.path.join(UPLOAD_FOLDER, img) for img in uploaded_images]
            audio_path = os.path.join(AUDIO_FOLDER, uploaded_audio)
            output_filename = f"video_{'zoom' if use_zoom else 'static'}_{uuid.uuid4().hex[:6]}.mp4"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            
            task_id = str(uuid.uuid4())
            session['task_id'] = task_id
            progress_store[task_id] = 0

            video_thread = Thread(
                target=video_generation_task, 
                args=(
                    image_paths, transcription, audio_path, output_path,
                    font_size, text_position, 60, segments, 
                    user_aspect_ratio, words_per_chunk, task_id, output_filename, 
                    use_zoom # Passed the new flag
                )
            )
            video_thread.start()

            return redirect(url_for('index')) 

    # GET request handler (and post-POST render)
    return render_template(
        'index.html',
        uploaded_images=uploaded_images,
        uploaded_audio=uploaded_audio,
        transcription=transcription,
        result_video=result_video,
        task_id=task_id, 
        progress_store=progress_store
    )

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
    task_id = session.get('task_id')
    # If a task exists, mark it as cancelled so background worker can stop gracefully.
    if task_id:
        progress_store[task_id] = -1
        progress_store[f"error_{task_id}"] = "Cancelled by user via reset."
        progress_store.pop(f"result_{task_id}", None)

    # Try to delete files but ignore files that are in-use or cause errors.
    for folder in [UPLOAD_FOLDER, AUDIO_FOLDER, OUTPUT_FOLDER]:
        try:
            for filename in os.listdir(folder):
                if filename == ".gitkeep":
                    continue
                path = os.path.join(folder, filename)
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        os.remove(path)
                except PermissionError:
                    # File is in use (common on Windows); skip it and log.
                    print(f"[RESET] PermissionError deleting {path}; skipping.", file=sys.stderr)
                except Exception as e:
                    print(f"[RESET] Error deleting {path}: {e}", file=sys.stderr)
        except FileNotFoundError:
            # Folder might not exist; ignore
            pass
        except Exception as e:
            print(f"[RESET] Error listing {folder}: {e}", file=sys.stderr)

    # Attempt to clean TMP_FOLDER (try/except to be safe)
    try:
        for entry in os.listdir(TMP_FOLDER):
            p = os.path.join(TMP_FOLDER, entry)
            try:
                if os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    os.remove(p)
            except Exception:
                pass
    except Exception:
        pass

    # Clear session and redirect to index
    session.clear()
    return redirect(url_for('index'))

@app.route('/progress/<task_id>')
def progress(task_id):
    def event_stream():
        # Wait for the task to start
        progress = progress_store.get(task_id, None)
        while progress is None:
            time.sleep(0.1) 
            progress = progress_store.get(task_id, None)
            
        # Main polling loop
        while progress < 100 and progress >= 0:
            yield f"data: {progress}\n\n"
            time.sleep(0.5)
            progress = progress_store.get(task_id, 100)
        
        # Final status check (100 or -1)
        final_progress = progress_store.get(task_id, 100) 
        yield f"data: {final_progress}\n\n"
        
    return Response(stream_with_context(event_stream()), mimetype='text/event-stream')


@app.route('/complete_task/<task_id>')
def complete_task(task_id):
    """
    Called by client-side JS when SSE stream closes (progress=100 or -1).
    This handles final server-side session cleanup and redirect.
    """
    result_filename = progress_store.get(f"result_{task_id}")
    
    if result_filename and progress_store.get(task_id, 0) == 100:
        session['result_video'] = result_filename
    
    # Always clear the task session variable regardless of success/failure
    session.pop('task_id', None) 
    
    # Clean up global progress store entries
    progress_store.pop(task_id, None)
    progress_store.pop(f"result_{task_id}", None)
    progress_store.pop(f"error_{task_id}", None)

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, threaded=True)