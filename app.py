import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from fer import FER
import av
import cv2

# Page settings
st.set_page_config(
    page_title="AI Stress Detector",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Stress & Emotion Detector")
st.write("Live emotion & stress detection using FER (Streamlit Cloud Compatible).")

# Stress scoring based on emotion
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

# Video processor class
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = FER()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        result = self.detector.detect_emotions(rgb_img)

        if result:
            emotion_dict = result[0]["emotions"]
            emotion = max(emotion_dict, key=emotion_dict.get)

            stress = stress_value(emotion)

            cv2.putText(img, f"Emotion: {emotion}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            cv2.putText(img, f"Stress: {stress}%", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Start webcam
webrtc_streamer(
    key="camera",
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={"video": True, "audio": False}
)
