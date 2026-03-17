import os
import google.generativeai as genai
import cv2
from PIL import Image
import numpy as np
import json
from dotenv import load_dotenv

load_dotenv()

# ==== Config ====
# Assuming your .env file has: GEMINI_API_KEY="your-key-here"
API_KEY = os.getenv("GEMINI_API_KEY")
VIDEO_FILE = "/teamspace/studios/this_studio/output_clips/clip_001.mp4"
FRAME_INTERVAL_SECONDS = 2  # extract a frame every 2 seconds

SAFETY_SETTINGS = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_NONE"
  },
]




# ==== Gemini Setup ====
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash-latest",generation_config={"response_mime_type": "application/json"},
                                     safety_settings=SAFETY_SETTINGS)

# ==== Frame Extraction ====
def extract_frames(video_path, interval_sec=2):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * interval_sec)
    frames = []

    frame_count = 0
    success = True
    while success:
        success, frame = cap.read()
        if not success:
            break
        if frame_count % interval == 0:
            # Convert to RGB and then to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            frames.append(img)
        frame_count += 1

    cap.release()
    return frames

# ==== Prepare prompt ====
def build_prompt():
    return (
        "You are an AI security analyst. Analyze the CCTV footage images and answer:\n"
        "- Were any criminals detected? (yes/no)\n"
        "- How many individuals appear hooded?\n"
        "- For each suspected criminal, give:\n"
        "   - An ID (e.g., Criminal 1, Criminal 2, ...)\n"
        "   - A description of their dress\n"
        "   - Whether they appear to be armed (yes/no)\n"
        "Respond in this exact JSON format:\n"
        "{\n"
        "  \"criminals_detected\": yes/no,\n"
        "  \"number_hooded_individuals\": int,\n"
        "  \"number_hostages\": int,\n"
        "  \"Crime_intensity_level\": low/medium/high],\n"
        "  \"criminals\": [\n"
        "    {\n"
        "      \"criminal_id\": str,\n"
        "      \"criminal_dress\": str,\n"
        "      \"criminal_armed\": yes/no\n"
        "    }\n"
        "  ],\n"
        "  \"incident_description\": Description of the incident,\n"
        "}"
    )
def validate_json(json_data):
    try:
        json_obj = json.loads(json_data)
        # Your validation logic here, e.g., checking specific keys and data types
        print("JSON is valid:")
        return json_obj
    except json.JSONDecodeError as e:
        print("Invalid JSON:", e)
        print(json_data)
        return None
        # raise e
        # Call your run_inference API to get a new result


# ==== Run Gemini API ====
def analyze_video_with_gemini(video_path):
    frames = extract_frames(video_path, FRAME_INTERVAL_SECONDS)
    prompt = build_prompt()

    # Only take up to 5 frames to keep it efficient
    limited_frames = frames
    response = model.generate_content([prompt] + limited_frames)
    response_json = validate_json(response.text)

    with open("post_process_description.json", "w") as f:
      print(f"Writing raw response to file: {f.name}")
      json.dump(response_json, f, indent=2)

# ==== Execute ====
result = analyze_video_with_gemini(VIDEO_FILE)
print(json.dumps(result, indent=2))
