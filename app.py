import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import numpy as np

st.set_page_config(page_title="AI Stress Detector", layout="wide")

st.title("🤖 AI Stress & Emotion Detector")
st.write("Live emotion & stress detection using HSEmotion (Cloud Compatible).")

# ------------------------ Load Model ------------------------ #
@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained("HSEmotion/hsemotion-classification")
    model = AutoModelForImageClassification.from_pretrained("HSEmotion/hsemotion-classification")
    return processor, model

processor, model = load_model()

# ------------------------ Stress Function ------------------------ #
def stress_value(emotion):
    stress_map = {
        "happy": 10,
        "neutral": 30,
        "sad": 60,
        "fear": 80,
        "angry": 90,
        "disgust": 85,
        "surprise": 40
    }
    return stress_map.get(emotion, 50)

# ------------------------ Video Processing ------------------------ #
class EmotionProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        inputs = processor(images=rgb, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        emotion_id = int(torch.argmax(probs))
        emotion = model.config.id2label[emotion_id].lower()

        stress = stress_value(emotion)

        cv2.putText(img, f"Emotion: {emotion}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv2.putText(img, f"Stress: {stress}%", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ------------------------ Start Webcam ------------------------ #
webrtc_streamer(
    key="camera",
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={"video": True, "audio": False}
)
