from datetime import timedelta
import os
from threading import Thread
import time
import uuid

from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    stream_with_context,
    url_for,
)
from werkzeug.utils import secure_filename
import torch
import whisper
from moviepy.editor import (
    AudioFileClip,
    ColorClip,
    CompositeVideoClip,
    ImageClip,
    concatenate_videoclips,
)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", str(uuid.uuid4()))
app.permanent_session_lifetime = timedelta(minutes=60)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_FOLDER = os.path.join(STATIC_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(STATIC_DIR, "outputs")
AUDIO_FOLDER = os.path.join(STATIC_DIR, "audio")

for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, AUDIO_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# In-memory state is acceptable for the current single-process prototype.
# A multi-worker deployment should replace this with shared durable state.
progress_store = {}

device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("base", device=device)

ASPECT_RATIOS = {
    "16:9": (1920, 1080),
    "1:1": (1080, 1080),
    "9:16": (1080, 1920),
}

ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
ALLOWED_AUDIO_EXTENSIONS = {"mp3", "wav", "m4a", "aac", "ogg", "flac"}


def allowed_extension(filename, allowed_extensions):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


def unique_storage_name(filename):
    safe_name = secure_filename(filename)
    if not safe_name:
        raise ValueError("Invalid filename")
    return f"{uuid.uuid4().hex[:12]}__{safe_name}"


def remove_if_present(folder, filename):
    if not filename:
        return
    path = os.path.join(folder, filename)
    if os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            pass


def fast_video_generation(
    image_paths,
    audio_path,
    output_path,
    aspect_ratio="16:9",
    use_zoom=False,
    task_id=None,
):
    audio_clip = None
    video = None
    clips = []

    try:
        audio_clip = AudioFileClip(audio_path)
        audio_duration = audio_clip.duration
        num_images = len(image_paths)
        image_duration = audio_duration / num_images
        frame_w, frame_h = ASPECT_RATIOS.get(aspect_ratio, ASPECT_RATIOS["16:9"])

        for index, image_path in enumerate(image_paths):
            if task_id and progress_store.get(task_id) == -1:
                return

            image_clip = ImageClip(image_path)
            image_w, image_h = image_clip.size
            scale = min(frame_w / image_w, frame_h / image_h)
            image_clip = image_clip.resize((int(image_w * scale), int(image_h * scale)))

            background = ColorClip(
                size=(frame_w, frame_h),
                color=(255, 255, 255),
            ).set_duration(image_duration)

            final_clip = CompositeVideoClip(
                [background, image_clip.set_position("center")]
            ).set_duration(image_duration)

            if use_zoom:
                final_clip = final_clip.resize(
                    lambda t: 1.0 + 0.15 * (t / image_duration)
                )

            clips.append(final_clip)

            if task_id:
                progress_store[task_id] = int((index + 1) / num_images * 80)

        video = concatenate_videoclips(clips, method="compose").set_audio(audio_clip)
        temp_audio_path = f"{output_path}.audio.m4a"
        video.write_videofile(
            output_path,
            fps=24,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=temp_audio_path,
            remove_temp=True,
            logger=None,
        )

        if task_id and progress_store.get(task_id) != -1:
            progress_store[task_id] = 100
            progress_store[f"result_{task_id}"] = os.path.basename(output_path)
    except Exception as exc:
        if task_id:
            progress_store[task_id] = -1
            progress_store[f"error_{task_id}"] = str(exc)
    finally:
        if video is not None:
            video.close()
        for clip in clips:
            clip.close()
        if audio_clip is not None:
            audio_clip.close()


def transcribe_audio(audio_path):
    try:
        result = model.transcribe(audio_path)
        return result.get("text", "")
    except Exception as exc:
        return f"Error transcribing audio: {exc}"


@app.route("/", methods=["GET", "POST"])
def index():
    session.permanent = True
    uploaded_images = session.get("uploaded_images", [])
    uploaded_audio = session.get("uploaded_audio")
    task_id = session.get("task_id")
    result_video = session.get("result_video")

    if request.method == "POST":
        if "images" in request.files:
            files = request.files.getlist("images")
            uploaded_images = session.get("uploaded_images", [])

            for file in files:
                if not file or not file.filename:
                    continue
                if not allowed_extension(file.filename, ALLOWED_IMAGE_EXTENSIONS):
                    abort(400, description="Unsupported image file type")

                filename = unique_storage_name(file.filename)
                file.save(os.path.join(UPLOAD_FOLDER, filename))
                uploaded_images.append(filename)

            session["uploaded_images"] = uploaded_images
            session.pop("result_video", None)
            session.pop("task_id", None)
            return redirect(url_for("index"))

        if "audio" in request.files:
            file = request.files["audio"]
            if file.filename:
                if not allowed_extension(file.filename, ALLOWED_AUDIO_EXTENSIONS):
                    abort(400, description="Unsupported audio file type")

                previous_audio = session.get("uploaded_audio")
                remove_if_present(AUDIO_FOLDER, previous_audio)

                filename = unique_storage_name(file.filename)
                audio_path = os.path.join(AUDIO_FOLDER, filename)
                file.save(audio_path)
                session["uploaded_audio"] = filename

                # Flask session is request-context-bound; keep this synchronous
                # rather than mutating session state from a detached thread.
                session["audio_transcript"] = transcribe_audio(audio_path)
                return redirect(url_for("index"))

        if "generate" in request.form and uploaded_images and uploaded_audio:
            video_settings = {
                "aspect_ratio": request.form.get("aspect_ratio", "16:9"),
                "use_zoom": "zoom_effect" in request.form,
            }
            if video_settings["aspect_ratio"] not in ASPECT_RATIOS:
                abort(400, description="Unsupported aspect ratio")
            session["video_settings"] = video_settings

            ordered_filenames = request.form.getlist("image_order[]")
            if ordered_filenames:
                if sorted(ordered_filenames) != sorted(uploaded_images):
                    abort(400, description="Invalid image order")
                selected_images = ordered_filenames
            else:
                selected_images = uploaded_images

            image_paths = [os.path.join(UPLOAD_FOLDER, name) for name in selected_images]
            audio_path = os.path.join(AUDIO_FOLDER, uploaded_audio)

            output_filename = (
                f"video_{'zoom' if video_settings['use_zoom'] else 'static'}_"
                f"{uuid.uuid4().hex[:10]}.mp4"
            )
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)

            task_id = str(uuid.uuid4())
            session["task_id"] = task_id
            progress_store[task_id] = 0

            Thread(
                target=fast_video_generation,
                args=(
                    image_paths,
                    audio_path,
                    output_path,
                    video_settings["aspect_ratio"],
                    video_settings["use_zoom"],
                    task_id,
                ),
                daemon=True,
            ).start()

            return redirect(url_for("index"))

    return render_template(
        "index.html",
        uploaded_images=uploaded_images,
        uploaded_audio=uploaded_audio,
        result_video=result_video,
        task_id=task_id,
        progress_store=progress_store,
        video_settings=session.get("video_settings", {}),
        audio_transcript=session.get(
            "audio_transcript",
            "Transcript will appear here after processing.",
        ),
    )


@app.route("/uploads/<filename>")
def send_uploaded(filename):
    if filename not in session.get("uploaded_images", []):
        abort(404)
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/outputs/<filename>")
def send_output(filename):
    if filename != session.get("result_video"):
        abort(404)
    return send_from_directory(OUTPUT_FOLDER, filename)


@app.route("/audio/<filename>")
def send_audio(filename):
    if filename != session.get("uploaded_audio"):
        abort(404)
    return send_from_directory(AUDIO_FOLDER, filename)


@app.route("/reset", methods=["POST"])
def reset():
    task_id = session.get("task_id")
    if task_id:
        progress_store[task_id] = -1
        progress_store[f"error_{task_id}"] = "Cancelled by user via reset."
        progress_store.pop(f"result_{task_id}", None)

    for filename in session.get("uploaded_images", []):
        remove_if_present(UPLOAD_FOLDER, filename)
    remove_if_present(AUDIO_FOLDER, session.get("uploaded_audio"))
    remove_if_present(OUTPUT_FOLDER, session.get("result_video"))

    session.clear()
    return redirect(url_for("index"))


@app.route("/progress/<task_id>")
def progress(task_id):
    if task_id != session.get("task_id"):
        abort(404)

    def event_stream():
        current = progress_store.get(task_id)
        while current is None:
            time.sleep(0.1)
            current = progress_store.get(task_id)

        while 0 <= current < 100:
            yield f"data: {current}\n\n"
            time.sleep(0.5)
            current = progress_store.get(task_id, 100)

        yield f"data: {progress_store.get(task_id, 100)}\n\n"

    return Response(stream_with_context(event_stream()), mimetype="text/event-stream")


@app.route("/complete_task/<task_id>", methods=["POST"])
def complete_task(task_id):
    if task_id != session.get("task_id"):
        abort(404)

    result_filename = progress_store.get(f"result_{task_id}")
    status = progress_store.get(task_id, 0)

    if result_filename and status == 100:
        previous_result = session.get("result_video")
        if previous_result and previous_result != result_filename:
            remove_if_present(OUTPUT_FOLDER, previous_result)
        session["result_video"] = result_filename
        payload = {"status": "ok", "filename": result_filename}
    else:
        payload = {
            "status": "error",
            "message": progress_store.get(f"error_{task_id}", "Unknown error"),
        }

    session.pop("task_id", None)
    progress_store.pop(task_id, None)
    progress_store.pop(f"result_{task_id}", None)
    progress_store.pop(f"error_{task_id}", None)
    return jsonify(payload)


if __name__ == "__main__":
    debug_enabled = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug_enabled, threaded=True, use_reloader=False)
