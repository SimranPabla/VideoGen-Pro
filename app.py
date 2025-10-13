from flask import Flask, render_template, request, Response, stream_with_context, send_from_directory, session, redirect, url_for, jsonify
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
import traceback
from multiprocessing import cpu_count
from moviepy.video.fx.fadein import fadein
from moviepy.video.fx.fadeout import fadeout

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
FONT_FOLDER = os.path.join(STATIC_DIR, 'fonts')
TMP_FOLDER = os.path.join(BASE_DIR, 'tmp_segments')
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, AUDIO_FOLDER, FONT_FOLDER, TMP_FOLDER]:
    os.makedirs(folder, exist_ok=True)

FONT_PATH = os.path.join(FONT_FOLDER, 'BebasNeue-Regular.ttf')

# --- Global Resources ---
FFMPEG_CODEC = "libx264"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("base", device=device)
progress_store = {}

# --- Aspect Ratios ---
ASPECT_RATIOS = {
    "16:9": (1920, 1080),
    "1:1": (1080, 1080),
    "9:16": (1080, 1920)
}

# --- Helper Functions ---
def create_static_overlay_clip(segment_text, font_size=50, text_position="bottom", canvas_size=(1920,1080), max_width=1720, duration=1.0):
    W,H = canvas_size
    padding = 20
    box_radius = 16
    box_alpha = 160
    text_color = (255,255,255,255)
    box_color = (0,0,0,box_alpha)

    overlay = Image.new("RGBA",(W,H),(0,0,0,0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype(FONT_PATH,font_size)
    except:
        font = ImageFont.load_default()

    lines=[]
    current=""
    words = segment_text.split()
    for w in words:
        test = w if current=="" else current+" "+w
        bbox = draw.textbbox((0,0),test,font=font)
        if bbox[2]-bbox[0]<=max_width:
            current=test
        else:
            if current: lines.append(current)
            current=w
    if current: lines.append(current)

    line_spacing=int(font_size*0.25)
    line_heights=[draw.textbbox((0,0),l,font=font)[3]-draw.textbbox((0,0),l,font=font)[1] for l in lines]
    line_height = max(line_heights) if line_heights else font_size
    text_block_w = max(draw.textbbox((0,0),l,font=font)[2]-draw.textbbox((0,0),l,font=font)[0] for l in lines) if lines else 0
    text_block_h = len(lines)*line_height + (len(lines)-1)*line_spacing

    box_w = text_block_w+2*padding
    box_h = text_block_h+2*padding
    if text_position=="top": box_y=50
    elif text_position=="center": box_y=(H-box_h)//2
    else: box_y=H-box_h-50
    box_x=(W-box_w)//2

    try:
        draw.rounded_rectangle([box_x,box_y,box_x+box_w,box_y+box_h],radius=box_radius,fill=box_color)
    except:
        draw.rectangle([box_x,box_y,box_x+box_w,box_y+box_h],fill=box_color)

    ty=box_y+padding
    for line in lines:
        line_w = draw.textbbox((0,0),line,font=font)[2]-draw.textbbox((0,0),line,font=font)[0]
        tx = box_x+padding+(box_w-2*padding-line_w)//2
        draw.text((tx,ty),line,font=font,fill=text_color)
        ty+=line_height+line_spacing

    overlay_array = np.array(overlay,dtype=np.uint8)
    clip = ImageClip(overlay_array).set_duration(duration)
    return clip

def create_image_base_clip(img_path,duration,aspect_ratio="16:9",zoom_start=1.0,zoom_end=1.15,use_zoom=True):
    frame_w,frame_h = ASPECT_RATIOS.get(aspect_ratio,(1920,1080))
    img = Image.open(img_path).convert("RGB")
    img_w,img_h = img.size
    target_ratio = frame_w/frame_h
    img_ratio = img_w/img_h
    if img_ratio>target_ratio:
        new_w = frame_w
        new_h = int(frame_w/img_ratio)
    else:
        new_h = frame_h
        new_w = int(frame_h*img_ratio)
    img_resized = img.resize((new_w,new_h), Image.Resampling.LANCZOS)
    bg = Image.new("RGB",(frame_w,frame_h),(255,255,255))
    x_offset = (frame_w-new_w)//2
    y_offset = (frame_h-new_h)//2
    bg.paste(img_resized,(x_offset,y_offset))
    padded_path = os.path.join(tempfile.gettempdir(),f"{uuid.uuid4().hex}.png")
    bg.save(padded_path)
    clip = ImageClip(padded_path).set_duration(duration)
    if use_zoom:
        clip = clip.resize(lambda t: zoom_start+(zoom_end-zoom_start)*(t/duration))
    clip = clip.set_position('center')
    return clip

def _ffmpeg_concat_entry(path):
    safe = path.replace("'","'\\''")
    return f"file '{safe}'\n"

def concat_segments_copy_video(segment_paths,audio_path,output_path,task_id=None):
    if task_id: progress_store[task_id]=85
    if not segment_paths: raise ValueError("No segments to concatenate")
    tmp_dir = os.path.dirname(segment_paths[0])
    list_file=os.path.join(tmp_dir,f"concat_{uuid.uuid4().hex}.txt")
    with open(list_file,"w",encoding="utf-8") as f:
        for p in segment_paths: f.write(_ffmpeg_concat_entry(p))
    cmd=["ffmpeg","-y","-hide_banner","-f","concat","-safe","0","-i",list_file,
         "-i",audio_path,"-map","0:v:0","-map","1:a:0",
         "-c:v","copy","-c:a","aac","-b:a","192k","-shortest",output_path]
    subprocess.run(cmd,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    os.remove(list_file)
    if task_id: progress_store[task_id]=95

# --- Background Task ---
def video_generation_task(image_paths, transcription, audio_path, output_path,
                          font_size, text_position, fps, segments, aspect_ratio,
                          words_per_chunk, task_id, output_filename, use_zoom, show_captions):
    tmp_dir = None
    try:
        audio_clip = AudioFileClip(audio_path)
        audio_duration = audio_clip.duration
        audio_clip.close()

        num_images = len(image_paths)
        if num_images == 0:
            raise ValueError("No images provided for video generation.")

        # --- Equal screen time per image ---
        seg_duration = audio_duration / num_images if num_images>0 else 1.0
        words = transcription.split() if transcription else []
        num_words = len(words)
        words_per_image = math.ceil(num_words / num_images) if num_words > 0 else 0

        caption_segments = []
        for i in range(num_images):
            start_word = i * words_per_image
            end_word = min(start_word + words_per_image, num_words)
            segment_words = words[start_word:end_word] if show_captions else []
            caption_segments.append({
                "words": segment_words,
                "duration": seg_duration
            })

        tmp_dir = tempfile.mkdtemp(dir=TMP_FOLDER)
        segment_paths = []

        for i, seg in enumerate(caption_segments):
            if task_id and progress_store.get(task_id) == -1:
                print(f"[INFO] Task {task_id} cancelled.")
                break

            img_path = image_paths[i % num_images]
            seg_filename = os.path.join(tmp_dir, f"segment_{i:06d}.mp4")

            base_clip = create_image_base_clip(img_path, seg["duration"], aspect_ratio, use_zoom=use_zoom)

            if show_captions and seg["words"]:
                text = " ".join(seg["words"])
                overlay_clip = create_static_overlay_clip(text, font_size, text_position,
                                                          (base_clip.w, base_clip.h), duration=seg["duration"])
                overlay_clip = overlay_clip.set_position("center")
                fade_duration = min(1.0, seg["duration"] * 0.2)
                overlay_clip = overlay_clip.fx(fadein, fade_duration).fx(fadeout, fade_duration)
                final_clip = CompositeVideoClip([base_clip, overlay_clip], size=(base_clip.w, base_clip.h))
            else:
                final_clip = base_clip

            final_clip = final_clip.set_duration(seg["duration"])
            final_clip.write_videofile(
                seg_filename,
                fps=fps,
                codec=FFMPEG_CODEC,
                audio=False,
                threads=1,
                preset="medium",
                logger=None,
                ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]
            )

            for clip in [base_clip, final_clip]:
                try: clip.close()
                except: pass

            segment_paths.append(seg_filename)

            if task_id and progress_store.get(task_id) != -1:
                progress_store[task_id] = int((i + 1) / max(1, num_images) * 80)

        if segment_paths:
            concat_segments_copy_video(segment_paths, audio_path, output_path, task_id)
            if task_id and progress_store.get(task_id) != -1:
                progress_store[task_id] = 100
                progress_store[f"result_{task_id}"] = output_filename
        else:
            raise ValueError("No segments were created. Check your images and transcription.")

    except Exception as e:
        if task_id:
            progress_store[task_id] = -1
            progress_store[f"error_{task_id}"] = str(e)
        print(f"[ERROR] Video generation failed: {e}\n{traceback.format_exc()}", file=sys.stderr)
    finally:
        if tmp_dir:
            try: shutil.rmtree(tmp_dir, ignore_errors=True)
            except: pass

# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    uploaded_images = session.get('uploaded_images', [])
    uploaded_audio = session.get('uploaded_audio', None)
    transcription = session.get('transcription', '')
    task_id = session.get('task_id', None)
    result_video = session.get('result_video', None)

    if request.method == 'POST':
        if 'images' in request.files:
            files = request.files.getlist('images')
            new_files = []
            for f in files:
                if f and f.filename != '':
                    filename = secure_filename(f.filename)
                    f.save(os.path.join(UPLOAD_FOLDER, filename))
                    new_files.append(filename)
            if new_files: session['uploaded_images'] = new_files
            session.pop('result_video', None)
            session.pop('task_id', None)
            return redirect(url_for('index'))

        if 'audio' in request.files:
            file = request.files['audio']
            if file.filename != '':
                filename = secure_filename(file.filename)
                audio_path = os.path.join(AUDIO_FOLDER, filename)
                file.save(audio_path)
                trans_task_id = str(uuid.uuid4())
                progress_store[trans_task_id] = 5
                try:
                    result = model.transcribe(audio_path, verbose=False)
                    session['uploaded_audio'] = filename
                    session['transcription'] = result.get('text', '')
                    progress_store[trans_task_id] = 100
                except:
                    progress_store[trans_task_id] = -1
                    session['uploaded_audio'] = None
                    session['transcription'] = "Transcription failed."
                return redirect(url_for('index'))

        if 'generate' in request.form and uploaded_images and uploaded_audio:
            video_settings = {
                'font_size': int(request.form.get('font_size', 50)),
                'text_position': request.form.get('text_position', 'bottom'),
                'aspect_ratio': request.form.get('aspect_ratio', '16:9'),
                'use_zoom': 'zoom_effect' in request.form,
                'show_captions': 'show_captions' in request.form,
                'words_per_chunk': int(request.form.get('words_per_chunk', 5))
            }
            session['video_settings'] = video_settings

            ordered_filenames = request.form.getlist('image_order[]')
            image_paths = [os.path.join(UPLOAD_FOLDER, f) for f in (ordered_filenames if ordered_filenames else uploaded_images)]
            audio_path = os.path.join(AUDIO_FOLDER, uploaded_audio)

            output_filename = f"video_{'zoom' if video_settings['use_zoom'] else 'static'}_{uuid.uuid4().hex[:6]}.mp4"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)

            task_id = str(uuid.uuid4())
            session['task_id'] = task_id
            progress_store[task_id] = 0

            video_thread = Thread(
                target=video_generation_task,
                args=(
                    image_paths, transcription, audio_path, output_path,
                    video_settings['font_size'], video_settings['text_position'],
                    60, None, video_settings['aspect_ratio'], video_settings['words_per_chunk'],
                    task_id, output_filename, video_settings['use_zoom'], video_settings['show_captions']
                )
            )
            video_thread.start()

            if 'regenerate' in request.form:
                return jsonify({"task_id": task_id})
            return redirect(url_for('index'))

    return render_template(
        'index.html',
        uploaded_images=uploaded_images,
        uploaded_audio=uploaded_audio,
        transcription=transcription,
        result_video=result_video,
        task_id=task_id,
        progress_store=progress_store,
        show_captions=session.get('video_settings', {}).get('show_captions', True),
        video_settings=session.get('video_settings', {})
    )

@app.route('/uploads/<filename>')
def send_uploaded(filename):
    return send_from_directory(UPLOAD_FOLDER,filename)

@app.route('/outputs/<filename>')
def send_output(filename):
    return send_from_directory(OUTPUT_FOLDER,filename)

@app.route('/audio/<filename>')
def send_audio(filename):
    return send_from_directory(AUDIO_FOLDER,filename)

@app.route('/reset')
def reset():
    task_id=session.get('task_id')
    if task_id:
        progress_store[task_id]=-1
        progress_store[f"error_{task_id}"]="Cancelled by user via reset."
        progress_store.pop(f"result_{task_id}",None)

    for folder in [UPLOAD_FOLDER,AUDIO_FOLDER,OUTPUT_FOLDER,TMP_FOLDER]:
        for f in os.listdir(folder):
            p=os.path.join(folder,f)
            try:
                if os.path.isdir(p): shutil.rmtree(p, ignore_errors=True)
                else: os.remove(p)
            except: pass

    session.clear()
    return redirect(url_for('index'))

@app.route('/progress/<task_id>')
def progress(task_id):
    def event_stream():
        progress = progress_store.get(task_id)
        while progress is None:
            time.sleep(0.1)
            progress=progress_store.get(task_id)
        while progress<100 and progress>=0:
            yield f"data: {progress}\n\n"
            time.sleep(0.5)
            progress=progress_store.get(task_id,100)
        yield f"data: {progress_store.get(task_id,100)}\n\n"
    return Response(stream_with_context(event_stream()),mimetype='text/event-stream')

@app.route('/complete_task/<task_id>')
def complete_task(task_id):
    result_filename=progress_store.get(f"result_{task_id}")
    status = progress_store.get(task_id, 0)

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
    app.run(debug=True,threaded=True)
