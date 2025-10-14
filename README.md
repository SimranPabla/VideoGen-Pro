# 🎥 VideoGen Pro ✨  
![HTML5](https://img.shields.io/badge/HTML5-orange?logo=html5)  ![CSS3](https://img.shields.io/badge/CSS3-blue?logo=css3) ![JavaScript](https://img.shields.io/badge/JavaScript-yellow?logo=javascript)  ![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)  

VideoGen Pro is a full-stack web application built on Flask that automates the creation of high-quality, professional videos from static images and an audio track.

The app provides a clean, step-by-step interface for users to upload assets, reorder images using drag-and-drop, and select options like dynamic zoom (Ken Burns effect) and aspect ratio (e.g., 9:16 vertical video).

On the backend, it uses OpenAI's Whisper for fast audio transcription and MoviePy/FFmpeg to seamlessly generate the final video with real-time progress feedback delivered via Server-Sent Events (SSE). It's a comprehensive tool designed to streamline the production of content like social media clips and narrated visual stories.

---

## 🚀 Features  
- 🖼️ **Drag & Drop Image Uploads**  
- ➕ **Add More Images** dynamically  
- 🧹 **Reset / Clear All** images  
- 🧭 **Smooth Preview Layout**  
- 💡 **Clean UI with modern design**

---

## 🧰 Tech Stack  
- **Frontend:** HTML5, CSS3, JavaScript
- **Backend:** Python, Flask, MoviePy
- **Framework:** Pure HTML/CSS/JS
- **Design:** Responsive, animated drag-and-drop zone  

---
# VideoGen Pro Screenshot

![VideoGen Pro Screenshot](https://raw.githubusercontent.com/SimranPabla/gifmagic.ai/refs/heads/main/templates/preview/preview1.png)

![VideoGen Pro Screenshot](https://raw.githubusercontent.com/SimranPabla/gifmagic.ai/refs/heads/main/templates/preview/preview2.png)

![VideoGen Pro Screenshot](https://raw.githubusercontent.com/SimranPabla/gifmagic.ai/refs/heads/main/templates/preview/preview3.png)

![VideoGen Pro Screenshot](https://raw.githubusercontent.com/SimranPabla/gifmagic.ai/refs/heads/main/templates/preview/preview4.png)

![VideoGen Pro Screenshot](https://raw.githubusercontent.com/SimranPabla/gifmagic.ai/refs/heads/main/templates/preview/preview5.png)

![VideoGen Pro Screenshot](https://raw.githubusercontent.com/SimranPabla/gifmagic.ai/refs/heads/main/templates/preview/preview6.png)

---

## 📄 Code Example  

Here’s the basic HTML structure for **VideoGen Pro**:

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VideoGen Pro ✨</title>
<style>
/* Add your styling here */
</style>
</head>
<body>
  <div class="container">
    <h1>🎥 VideoGen Pro ✨</h1>
    <div id="drop-area" class="drop-zone">
      <p>Drag & Drop Images Here</p>
      <input type="file" id="fileElem" multiple accept="image/*">
      <button id="addMoreBtn">Add More</button>
      <div id="gallery"></div>
      <button id="resetBtn">Reset</button>
    </div>
  </div>

<script>
  // JavaScript logic for upload & preview
</script>
</body>
</html>
```
## 🛠️ Installation and Setup

### Prerequisites

1.  **Python 3.8+**
2.  **FFmpeg:** The `ffmpeg` command-line tool **must** be installed on your system and accessible via your system's PATH. This is essential for all video encoding operations.
3.  **PyTorch & CUDA (Recommended):** For the fastest transcription performance, a system with a **CUDA-enabled GPU** and the appropriate PyTorch version is highly recommended. The application defaults gracefully to CPU if no GPU is found.

### Steps

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/YourUsername/videogen-pro.git
    cd videogen-pro
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    .\venv\Scripts\activate   # On Windows
    ```

3.  **Install dependencies:**
    You'll need a `requirements.txt` file (not provided, but inferred from the code).

    ```bash
    # Assuming you have a requirements.txt with flask, whisper, torch, moviepy, pillow, werkzeug, etc.
    pip install -r requirements.txt

    # If using GPU, ensure you install torch with CUDA support first!
    # pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuXX
    # pip install -U openai-whisper moviepy Pillow Flask
    ```

4.  **Set a Secret Key (Optional but Recommended):**
    For production, you should set a secure secret key. It defaults to a generated UUID if not set.

    ```bash
    export FLASK_SECRET_KEY="your_very_secure_secret_key"
    ```

5.  **Run the application:**

    ```bash
    python your_app_file_name.py  # Replace with the actual file name (e.g., app.py)
    ```

The application will start, usually accessible at `http://127.0.0.1:5000/`.

-----

## ⚙️ How It Works

1.  **Upload Images (Step 1):** Users upload images. These are saved to `static/uploads`. Drag-and-drop powered by **SortableJS** allows for quick reordering.
2.  **Upload Audio (Step 2):** Users upload an audio file (MP3, WAV, etc.). This triggers a separate thread to run the **Whisper** transcription model, and the result is stored in the session.
3.  **Generate Video (Step 3):**
      * The user selects options (Aspect Ratio, Zoom Effect).
      * A unique `task_id` is created, and a new thread starts the `fast_video_generation` function.
      * The browser connects to `/progress/<task_id>` using **Server-Sent Events (SSE)** to get real-time progress updates.
      * The video generation process loops through the ordered images, assigning each image an equal duration based on the audio length.
      * The final clips are concatenated, and the original audio track is added.
      * Once complete, the client-side JavaScript receives the `100%` signal and requests the final video details from `/complete_task/<task_id>`.

## 📂 Project Structure

```
.
├── static/
│   ├── uploads/     # Stores uploaded images
│   ├── audio/       # Stores uploaded audio
│   ├── outputs/     # Stores final generated videos
│   └── fonts/       # (Implied/missing but often used) Stores custom fonts
├── tmp_segments/    # Used for temporary files during moviepy/ffmpeg processing
├── templates/
│   └── index.html   # The main (and only) HTML template
└── your_app_file_name.py  # The main Flask application file
```

-----

## 🌐 Roadmap & Potential Improvements

  * **Segmented Captioning:** Currently, transcription is used for display only. A major upgrade would be to use the detailed Whisper segments for timed, accurate burn-in captions that change with the spoken word.
  * **Cleanup Service:** Implement a scheduled task to periodically clean out old files from `uploads`, `audio`, and `outputs` to manage disk space.
  * **Custom Fonts/Styling:** Allow users to upload or select different font styles and caption colors.
  * **Dockerization:** Provide a `Dockerfile` for easy deployment in containerized environments.

-----

# 🧑‍💻 Author

### Simranjit Singh 
* **📍 Edmonton, Alberta**
* **💬 Passionate about AI, cybersecurity, and creative tech solutions.**

## 🪪 License
This project is open-source and available under the MIT License.
