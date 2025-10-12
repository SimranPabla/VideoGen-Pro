from flask import Flask, render_template, request, send_from_directory, session, redirect, url_for
import os
from werkzeug.utils import secure_filename
from datetime import timedelta
import whisper
import torch
from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ThreadPoolExecutor, as_completed
from moviepy.editor import ImageClip, AudioFileClip, CompositeVideoClip
from moviepy.video.fx.fadein import fadein
from moviepy.video.fx.fadeout import fadeout
import subprocess
import tempfile
import shutil
import uuid
import numpy as np
from collections import defaultdict
import time
import sys

# --- Tunables ---
CAPTION_DELAY_SECONDS = 0.0     # captions start offset within each segment (0 for immediate)
ZOOM_MAX_DELTA = 0.15           # final zoom amount (e.g., 0.08 -> 8%)
ZOOM_PORTION = 0.7              # fraction of the segment duration used to reach final zoom
MIN_CAPTION_DURATION = 0.1      # guard minimum caption display length
SEGMENT_MIN_DURATION = 0.5      # minimum duration assigned to a segment to avoid too short clips
MAX_WORKERS = 4                 # cap thread workers

# --- Flask setup ---
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev_insecure_secret_key")
app.permanent_session_lifetime = timedelta(minutes=30)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB uploads

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

# --- Detect GPU support for FFmpeg ---
def has_nvenc():
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return "h264_nvenc" in result.stdout
    except Exception:
        return False

FFMPEG_CODEC = "h264_nvenc" if has_nvenc() else "libx264"

# --- Load Whisper model with GPU support ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[startup] Whisper device: {device}", file=sys.stderr)
model = whisper.load_model("base", device=device)
print("[startup] Whisper model loaded", file=sys.stderr)

# --- Preprocess images to 1080p with padding ---
def preprocess_images(image_paths):
    """Pads all images to 1920x1080 with white background while preserving original size."""
    target_size = (1920, 1080)
    padded_paths = []

    for idx, path in enumerate(image_paths):
        img = Image.open(path).convert("RGB")
        img.thumbnail(target_size, Image.Resampling.LANCZOS)

        bg = Image.new("RGB", target_size, (255, 255, 255))
        x_offset = (target_size[0] - img.width) // 2
        y_offset = (target_size[1] - img.height) // 2
        bg.paste(img, (x_offset, y_offset))

        save_path = os.path.join(UPLOAD_FOLDER, f"padded_{idx}.png")
        bg.save(save_path)
        padded_paths.append(save_path)

    return padded_paths

# --- Text overlay using Pillow (no ImageMagick) ---
def build_text_overlay_rgba(text, font_size, position, canvas_size=(1920, 1080), max_width=1720):
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

    # Word-wrap text
    words = (text or "").split()
    lines = []
    current = ""
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

    if not lines:
        return np.zeros((H, W, 4), dtype=np.uint8)

    line_heights, line_widths = [], []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])
    line_height = max(line_heights) if line_heights else font_size
    text_block_w = min(max(line_widths), max_width)
    text_block_h = len(lines) * line_height + (len(lines)-1) * int(line_height * 0.25)

    box_w = text_block_w + 2 * padding
    box_h = text_block_h + 2 * padding

    if position == "top":
        box_y = 50
    elif position == "center":
        box_y = (H - box_h) // 2
    else:
        box_y = H - box_h - 50
    box_x = (W - box_w) // 2

    try:
        draw.rounded_rectangle([box_x, box_y, box_x+box_w, box_y+box_h], radius=box_radius, fill=box_color)
    except Exception:
        draw.rectangle([box_x, box_y, box_x+box_w, box_y+box_h], fill=box_color)

    tx, ty = box_x + padding, box_y + padding
    for line in lines:
        draw.text((tx, ty), line, font=font, fill=text_color)
        ty += line_height + int(line_height * 0.25)

    return np.array(overlay, dtype=np.uint8)

# --- Zoom scaling (zoom happens once, then holds) ---
def zoom_scale_smoothstep(t, duration, max_delta=ZOOM_MAX_DELTA, portion=ZOOM_PORTION):
    """
    Smoothstep zoom that reaches final scale within 'portion' of duration,
    then holds steady at the final zoom level.
    """
    if portion <= 0:
        portion = 1.0
    zoom_time = max(duration * portion, 1e-6)
    if t >= zoom_time:
        return 1.0 + max_delta  # hold final zoom after initial zoom
    u = t / zoom_time
    s = u * u * (3 - 2 * u)  # smoothstep easing
    return 1.0 + max_delta * s

# --- Create video segment (now supports one-time zoom per image) ---
def create_segment_file(index, img_path, text, duration, font_size, text_position, tmp_dir, fps, enable_zoom=True):
    seg_filename = os.path.join(tmp_dir, f"segment_{index:06d}.mp4")

    start_time = time.time()
    print(f"[segment {index}] start rendering (duration={duration:.2f}s) -> {seg_filename}", file=sys.stderr)

    base = ImageClip(img_path).set_duration(duration)

    if enable_zoom:
        zoomed = base.resize(lambda t: zoom_scale_smoothstep(t, duration)).set_position("center")
    else:
        zoomed = base.set_position("center")

    rgba = build_text_overlay_rgba(text=text, font_size=font_size, position=text_position, canvas_size=(1920,1080))
    text_clip = None
    if rgba.size != 0 and rgba.shape[2] == 4:
        rgb = rgba[:, :, :3]
        alpha = rgba[:, :, 3].astype(np.float32)/255.0

        # Caption start within segment (use configured CAPTION_DELAY_SECONDS but clamp so caption isn't too short)
        caption_start = min(CAPTION_DELAY_SECONDS, max(0.0, duration * 0.5))
        caption_duration = max(MIN_CAPTION_DURATION, duration - caption_start)

        text_clip = ImageClip(rgb).set_duration(caption_duration)
        mask_clip = ImageClip(alpha, ismask=True).set_duration(caption_duration)
        text_clip = text_clip.set_mask(mask_clip)

        if caption_start > 0:
            text_clip = text_clip.set_start(caption_start)

        fade_time = min(0.3, caption_duration / 3)
        if fade_time > 0:
            text_clip = fadein(text_clip, fade_time)
            text_clip = fadeout(text_clip, fade_time)

    clips = [zoomed]
    if text_clip is not None:
        clips.append(text_clip)

    composite = CompositeVideoClip(clips, size=(1920,1080)).set_duration(duration)

    try:
        # show ffmpeg progress in terminal (verbose True)
        composite.write_videofile(
            seg_filename,
            fps=fps,
            codec="libx264",
            audio=False,
            threads=1,
            preset="fast",
            verbose=True,
            logger=None,  # None will print progress to console
            ffmpeg_params=["-pix_fmt","yuv420p"]
        )
    finally:
        for clip in [composite, base, zoomed, text_clip]:
            if clip is not None:
                try:
                    clip.close()
                except Exception:
                    pass

    elapsed = time.time() - start_time
    print(f"[segment {index}] finished in {elapsed:.2f}s -> {seg_filename}", file=sys.stderr)
    return seg_filename

# --- Safe ffmpeg concat entry writer ---
def _ffmpeg_concat_entry(path):
    # Escape single quotes for ffmpeg concat demuxer
    escaped = path.replace("'", "'\\''")
    return f"file '{escaped}'\n"

def concat_segments_copy_video(segment_paths, audio_path, output_path):
    if not segment_paths:
        raise ValueError("No segments to concatenate")

    tmp_dir = os.path.dirname(segment_paths[0])
    list_file = os.path.join(tmp_dir, f"concat_{uuid.uuid4().hex}.txt")
    with open(list_file, "w", encoding="utf-8") as f:
        for p in segment_paths:
            f.write(_ffmpeg_concat_entry(p))

    cmd = [
        "ffmpeg", "-y", "-hide_banner",
        "-f", "concat", "-safe", "0", "-i", list_file,
        "-i", audio_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        output_path
    ]
    print(f"[concat] running ffmpeg concat -> {output_path}", file=sys.stderr)
    start = time.time()
    subprocess.run(cmd, check=True)
    print(f"[concat] finished in {(time.time()-start):.2f}s", file=sys.stderr)

# --- Generate video in parallel (with per-segment zoom control) ---
def generate_video_parallel(image_paths, transcription, audio_path, output_path,
                            font_size=50, text_position='bottom', fps=60, segments=None):
    start_all = time.time()
    print("[generate] starting video generation", file=sys.stderr)

    audio_clip = AudioFileClip(audio_path)
    try:
        audio_duration = audio_clip.duration
    finally:
        audio_clip.close()

    # --- use timestamps if provided ---
    caption_segments = []
    if segments and len(segments) > 0:
        for seg in segments:
            # Whisper segment keys are floats or strings - normalize
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start + 1.0))
            text = seg.get("text", "").strip()
            if not text:
                continue
            # enforce minimum duration
            if end - start < SEGMENT_MIN_DURATION:
                end = start + SEGMENT_MIN_DURATION
            caption_segments.append((start, end, text))
    else:
        # fallback if no timestamps: split by fixed groups (less ideal)
        words = (transcription or "").split()
        group_size = 5
        word_groups = [' '.join(words[i:i+group_size]) for i in range(0, len(words), group_size)] if words else [""]
        chunk_dur = max(audio_duration / max(1, len(word_groups)), SEGMENT_MIN_DURATION)
        t = 0.0
        for text in word_groups:
            start = t
            end = min(audio_duration, t + chunk_dur)
            caption_segments.append((start, end, text))
            t += chunk_dur

    num_segments = len(caption_segments)
    num_images = len(image_paths)
    if num_segments == 0:
        raise ValueError("No caption segments to render")

    # map segments to images (round-robin / proportional mapping)
    image_for_segment = [
        image_paths[min(int(i * num_images / max(1, num_segments)), num_images - 1)]
        for i in range(num_segments)
    ]

    # enable zoom only for first use of each image
    usage = defaultdict(int)
    enable_zoom_list = []
    for img in image_for_segment:
        enable_zoom_list.append(usage[img] == 0)
        usage[img] += 1

    tmp_dir = tempfile.mkdtemp(dir=TMP_FOLDER)
    print(f"[generate] temp dir: {tmp_dir}", file=sys.stderr)

    max_workers = min((os.cpu_count() or 1), MAX_WORKERS)
    futures = {}
    segment_paths = [None] * num_segments

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        for i, (start, end, text) in enumerate(caption_segments):
            duration = max(SEGMENT_MIN_DURATION, end - start)
            # Pass caption text and full segment duration; create_segment_file will manage caption timing
            fut = exe.submit(
                create_segment_file,
                i,
                image_for_segment[i],
                text,
                duration,
                font_size,
                text_position,
                tmp_dir,
                fps,
                enable_zoom_list[i]
            )
            futures[fut] = i

        for fut in as_completed(futures):
            idx = futures[fut]
            seg_path = fut.result()
            segment_paths[idx] = seg_path
            print(f"[generate] segment {idx} ready: {seg_path}", file=sys.stderr)

    # Ensure in order and drop None
    segment_paths = [p for p in segment_paths if p]

    try:
        concat_segments_copy_video(segment_paths, audio_path, output_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    total_elapsed = time.time() - start_all
    print(f"[generate] finished total in {total_elapsed:.2f}s -> {output_path}", file=sys.stderr)


# --- Flask routes ---
@app.route('/', methods=['GET','POST'])
def index():
    uploaded_images = session.get('uploaded_images', [])
    uploaded_audio = session.get('uploaded_audio', None)
    transcription = session.get('transcription', '')

    font_size = int(request.form.get('font_size', 50))
    text_position = request.form.get('text_position', 'bottom')

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

        # Upload and transcribe audio
        if 'audio' in request.files:
            file = request.files['audio']
            if file and file.filename != '':
                filename = secure_filename(file.filename)
                audio_path = os.path.join(AUDIO_FOLDER, filename)
                file.save(audio_path)

                t0 = time.time()
                print(f"[transcribe] starting transcription -> {audio_path}", file=sys.stderr)
                result = model.transcribe(audio_path)   # segments included by default
                elapsed = time.time() - t0
                print(f"[transcribe] finished in {elapsed:.2f}s", file=sys.stderr)

                transcription = result.get('text', '')
                segments = result.get('segments', [])

                session['uploaded_audio'] = filename
                session['transcription'] = transcription
                session['segments'] = segments   # save timestamps

            return redirect(url_for('index'))

        # Generate video
        if request.form.get('effect') == 'zoom' and uploaded_images and (transcription is not None):
            image_paths = [os.path.join(UPLOAD_FOLDER, img) for img in uploaded_images]
            image_paths = preprocess_images(image_paths)
            audio_path = os.path.join(AUDIO_FOLDER, uploaded_audio)
            output_filename = "zoom_video.mp4"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)

            t0 = time.time()
            print("[route] starting generate_video_parallel()", file=sys.stderr)
            generate_video_parallel(
                image_paths,
                transcription,
                audio_path,
                output_path,
                font_size=font_size,
                text_position=text_position,
                fps=60,
                segments=session.get('segments', [])
            )
            print(f"[route] generate_video_parallel done in {(time.time()-t0):.2f}s", file=sys.stderr)

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
    app.run(debug=True, use_reloader=False)
