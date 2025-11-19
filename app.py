import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from deepface import DeepFace
import av
import cv2

# -------------------- PAGE SETTINGS -------------------- #
st.set_page_config(
    page_title="AI Stress Detector",
    page_icon="🤖",
    layout="wide"
)

# School + Team details
SCHOOL_NAME = "📚 Your School Name Here"
TEAM_MEMBERS = ["Tushar Agarwal", "Member 2", "Member 3", "Member 4", "Member 5"]

# -------------------- HEADER DESIGN -------------------- #
st.markdown(
    f"""
    <div style='background:#2B65EC;padding:20px;border-radius:10px;margin-bottom:20px;'>
        <h1 style='text-align:center;color:white;'>🤖 AI STRESS & EMOTION DETECTOR</h1>
        <h3 style='text-align:center;color:#e6e6e6;'>{SCHOOL_NAME}</h3>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------- TEAM INFO -------------------- #
with st.sidebar:
    st.markdown(
        "<h2>👥 Project Team</h2>",
        unsafe_allow_html=True
    )
    for name in TEAM_MEMBERS:
        st.write("• " + name)

    st.markdown("---")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/Iconic_image.png/240px-Iconic_image.png",
             caption="School Logo", use_column_width=True)

st.write("### 🎥 Live Emotion Detection with Stress Score")
st.write("This tool analyzes your face in real-time and calculates a stress percentage.")

# -------------------- STRESS SCORE FUNCTION -------------------- #
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

# -------------------- VIDEO PROCESSOR -------------------- #
class EmotionProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        try:
            result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
            emotion = result[0]["dominant_emotion"]
            stress = stress_value(emotion)

            cv2.putText(img, f"Emotion: {emotion}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(img, f"Stress: {stress}%", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        except:
            pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# -------------------- START CAMERA -------------------- #
webrtc_streamer(
    key="camera",
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={"video": True, "audio": False}
)
