# VideoGen Pro

Flask-based media-processing application that turns ordered images and an audio track into a narrated MP4 video, with Whisper transcription and Server-Sent Events (SSE) progress reporting.

The project combines a browser UI with a Python media-processing backend. It is designed as a **local/single-process prototype**, not as a production multi-user video service.

## Problem

Creating a narrated image video manually requires several repetitive steps: ordering visual assets, matching them to an audio track, normalizing the output frame size, rendering the video, and tracking long-running encoding work.

VideoGen Pro packages those steps into one workflow:

1. Upload images.
2. Reorder them in the browser.
3. Upload an audio track.
4. Transcribe the audio with Whisper.
5. Select an aspect ratio and optional zoom effect.
6. Render the final MP4 with MoviePy/FFmpeg.
7. Track generation progress in the browser through SSE.

## Architecture

```text
Browser
  |
  | image/audio uploads + settings
  v
Flask application
  |
  +--> session-bound asset metadata
  |
  +--> Whisper "base" model
  |      |
  |      +--> audio transcript
  |
  +--> background video-generation thread
         |
         +--> MoviePy composition
         +--> FFmpeg H.264/AAC encoding
         +--> in-memory task progress
                 |
                 v
              SSE stream
                 |
                 v
              Browser UI
```

## Current Features

- Multi-image upload
- Drag-and-drop image ordering with SortableJS
- Audio upload and Whisper transcription
- `16:9`, `1:1`, and `9:16` output ratios
- Optional dynamic zoom effect
- H.264 video with AAC audio
- Background video rendering
- Server-Sent Events progress updates
- Session-scoped access to uploaded and generated assets
- 200 MB Flask request-size limit

## Implementation

### Image composition

Each image is resized to fit inside the selected output frame while preserving its aspect ratio. The image is centered on a white background rather than stretched to fill the frame.

The audio duration is divided evenly across the ordered images, then MoviePy concatenates the clips and attaches the original audio track.

### Audio transcription

The application loads OpenAI Whisper's `base` model and selects CUDA when PyTorch reports that a compatible GPU is available; otherwise it runs on CPU.

Transcription is currently performed synchronously during audio upload. This is deliberate: Flask's cookie-backed `session` object is request-context-bound and should not be mutated from a detached background thread.

### Background video generation

Video rendering remains asynchronous. A background thread updates an in-memory task record while the browser consumes progress through `/progress/<task_id>` using Server-Sent Events.

This keeps the page responsive while MoviePy/FFmpeg performs encoding.

### File handling

Uploaded filenames are normalized with `secure_filename` and stored with a random prefix to reduce filename collisions.

The current branch also restricts accepted extensions:

- Images: `png`, `jpg`, `jpeg`, `webp`
- Audio: `mp3`, `wav`, `m4a`, `aac`, `ogg`, `flac`

Asset-serving routes verify that the requested file belongs to the active Flask session. Reset removes only files associated with the current session rather than clearing the shared runtime directories.

### Temporary encoding files

Each render uses an output-specific temporary audio filename instead of a single global `temp-audio.m4a`, preventing concurrent render jobs from targeting the same temporary path.

## Technology Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| Transcription | OpenAI Whisper, PyTorch |
| Video processing | MoviePy, FFmpeg |
| Frontend | HTML, CSS, JavaScript |
| Drag-and-drop ordering | SortableJS |
| Progress transport | Server-Sent Events (SSE) |
| Video encoding | H.264 (`libx264`) + AAC |

## Setup

### Prerequisites

- Python 3.9+
- `pip`
- FFmpeg available on the system `PATH`

Whisper invokes FFmpeg for media decoding, so FFmpeg must be installed separately.

A CUDA-capable PyTorch installation is optional. CPU execution is supported but transcription and rendering may be substantially slower.

### 1. Clone the repository

```bash
git clone https://github.com/SimranPabla/VideoGen-Pro.git
cd VideoGen-Pro
```

### 2. Create a virtual environment

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

`moviepy<2` is specified because this code uses the MoviePy 1.x `moviepy.editor` API.

### 4. Set a Flask secret key

For a stable local session across process restarts, set `FLASK_SECRET_KEY` before starting the application.

Linux/macOS:

```bash
export FLASK_SECRET_KEY="replace-with-a-random-secret"
```

Windows PowerShell:

```powershell
$env:FLASK_SECRET_KEY="replace-with-a-random-secret"
```

If the variable is omitted, the application generates an ephemeral key at startup; existing browser sessions will therefore become invalid when the process restarts.

### 5. Run

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000/
```

The first startup may take longer because Whisper may need to download the `base` model weights.

## Runtime Directories

The application creates these directories at runtime:

```text
static/
├── uploads/
├── audio/
└── outputs/
```

They are excluded from Git by `.gitignore` because they contain generated/user-supplied runtime data.

## Reliability and Security Considerations

Implemented safeguards include:

- `secure_filename` normalization
- random storage prefixes to reduce upload collisions
- explicit image/audio extension allowlists
- validation that submitted image ordering matches the active session's uploaded files
- session checks before serving uploaded audio/images or generated videos
- session-specific cleanup during reset
- unique temporary audio paths per render
- a 200 MB request-size limit
- configurable Flask secret key

These controls improve the local prototype, but they do **not** make it a production multi-user service.

## Limitations

- Task progress is stored in a Python dictionary, so state is lost on process restart.
- The progress store is not shared across multiple Flask workers or hosts.
- Background rendering uses Python threads rather than a durable job queue.
- Whisper transcription runs synchronously during audio upload and can block the request for large files or CPU-only systems.
- There is no authentication or user-account model.
- Runtime files from abandoned sessions are not removed by a scheduled cleanup service.
- File extension checks do not perform full content-type or media-format validation.
- There is currently no automated test suite.
- The application has not been documented here as production-ready or horizontally scalable.

## Engineering Decisions

### Keep long-running rendering out of the request path

Encoding is the longest application operation, so it is moved to a background thread and observed through SSE.

### Keep progress transport simple

SSE is sufficient for one-way progress updates from server to browser and avoids introducing WebSocket infrastructure for this prototype.

### Prefer honest single-process semantics

The application explicitly documents its in-memory progress state rather than implying that task state survives restarts or supports multiple workers.

### Protect session boundaries before adding scale

Asset routes and reset behavior are scoped to the active session so one browser session does not intentionally clear or retrieve another session's known files. A production version should go further and use per-user storage namespaces plus authentication/authorization.

## Demo

The repository previously referenced screenshots hosted in a different project. Those links were removed because they did not provide reliable evidence of this repository's current UI.

For now, the supported demo path is to run the application locally using the setup steps above.

## Future Work

- Add unit and integration tests for upload, ordering, task lifecycle, and file authorization
- Move rendering/transcription jobs to a durable worker queue
- Store task state in Redis or another shared backend
- Add scheduled cleanup for abandoned runtime files
- Add stronger media validation beyond filename extensions
- Add per-user storage namespaces and authentication if deployed for multiple users
- Add timed Whisper segment captions to generated videos
- Add Docker packaging and reproducible deployment configuration
- Add structured application logging and task-level diagnostics

## Project Status

VideoGen Pro is an engineering prototype demonstrating Flask application design, asynchronous media processing, Whisper integration, MoviePy/FFmpeg composition, SSE progress reporting, and practical file/session lifecycle handling.

It should be evaluated as a prototype with documented limitations, not as a production video platform.
