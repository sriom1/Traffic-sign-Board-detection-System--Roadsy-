# ROADSY - Streamlit app with Image, Video and **Webcam** support
# ---------------------------------------------------------------
# Requirements (run once):
#   pip install streamlit streamlit-webrtc ultralytics opencv-python-headless numpy pandas

import cv2
import pandas as pd
import streamlit as st
from ultralytics import YOLO
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import os

# ───────────────────────── UI heading & background ─────────────────────────
st.markdown(
    "<h1 style='color:white;'>ROADSY: Automated Traffic Sign Detection and Classification</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"]{
        background-image:url("https://github.com/srishanmukhaom/roadsy/blob/b8f83ad16897dd668efbb8a29bb4475d904312cc/wp3504268.jpg?raw=true");
        background-size:cover;
    }
    [data-testid="stHeader"]{background-color:rgba(0,0,0,0);}
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────  load model & labels (GLOBAL) ────────────────────────
@st.cache_resource
def load_model():
    """Load YOLO model once and cache it"""
    try:
        model = YOLO("best.pt")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_class_list():
    """Load class labels once and cache them"""
    try:
        if os.path.exists("roadsylabels.txt"):
            with open("roadsylabels.txt") as f:
                return f.read().splitlines()
        else:
            st.warning("roadsylabels.txt not found. Using default labels.")
            return [f"class_{i}" for i in range(80)]  # fallback
    except Exception as e:
        st.error(f"Error loading labels: {e}")
        return [f"class_{i}" for i in range(80)]

# Load global variables
model = load_model()
class_list = load_class_list()

# ─────────────────────────── helper for drawing ────────────────────────────
def annotate(frame):
    """run YOLO and draw boxes; returns BGR image"""
    if model is None:
        return frame
    
    try:
        results = model.predict(frame, verbose=False, conf=0.5)  # Added confidence threshold
        if not results or len(results) == 0:
            return frame
        
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return frame
            
        boxes_data = boxes.data.cpu().numpy()  # x1,y1,x2,y2,conf,class
        
        for box in boxes_data:
            x1, y1, x2, y2, conf, cls = box
            if conf > 0.5:  # Only show high confidence detections
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
                
                # Get class name safely
                class_name = class_list[int(cls)] if int(cls) < len(class_list) else f"class_{int(cls)}"
                label = f"{class_name} {conf:.2f}"
                
                cv2.putText(
                    frame,
                    label,
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255,0,0),
                    2,
                    cv2.LINE_AA,
                )
    except Exception as e:
        st.error(f"Error in annotation: {e}")
        
    return frame

# ──────────────────────── streamlit-webrtc class ───────────────────────────
class SignDetector(VideoTransformerBase):
    def __init__(self):
        self.model = model  # Use global model
        self.class_list = class_list  # Use global class list
        
    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            img = frame.to_ndarray(format="bgr24")
            # Optionally resize for better performance
            height, width = img.shape[:2]
            if width > 640:  # Resize if too large
                scale = 640 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = cv2.resize(img, (new_width, new_height))
            
            img = annotate(img)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception as e:
            st.error(f"Error in video transformation: {e}")
            return frame

# ───────────────────────────── WebRTC Configuration ─────────────────────────
# Configure ICE servers for better connectivity
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        # Add TURN servers if needed for production
    ]
})

# ───────────────────────────── main selector ─────────────────────────────
option = st.selectbox("Select detection method", ("Image Upload", "Webcam", "Video Upload"))

# 1️⃣ IMAGE ---------------------------------------------------------------
if option == "Image Upload":
    up = st.file_uploader("Choose an image…", type=["jpg","jpeg","png"])
    if up:
        try:
            img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
            img = annotate(img)
            st.image(img, channels="BGR", caption="Detected Objects", use_column_width=True)
        except Exception as e:
            st.error(f"Error processing image: {e}")

# 2️⃣ WEBCAM --------------------------------------------------------------
elif option == "Webcam":
    st.write("**Instructions:**")
    st.write("1. Click 'START' below to begin webcam stream")
    st.write("2. Allow camera access when prompted by your browser")
    st.write("3. If the stream doesn't start, check your browser permissions")
    
    # Check if deployed on Streamlit Cloud
    if 'share.streamlit.io' in st.get_option('browser.serverAddress') if st.get_option('browser.serverAddress') else False:
        st.warning("⚠️ **Deployment Note:** Webcam may not work on Streamlit Cloud due to server-side camera access limitations. Consider deploying on platforms that support WebRTC.")
    
    webrtc_streamer(
        key="roadsy-webcam",
        video_transformer_factory=SignDetector,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": {
                "width": {"min": 640, "ideal": 1280, "max": 1920},
                "height": {"min": 480, "ideal": 720, "max": 1080},
                "frameRate": {"min": 15, "ideal": 30, "max": 30}
            },
            "audio": False
        },
        async_processing=True,  # Better performance
    )

# 3️⃣ VIDEO FILE ----------------------------------------------------------
elif option == "Video Upload":
    up = st.file_uploader("Choose a video…", type=["mp4","mov","avi","mkv"])
    if up:
        try:
            # Save to temp file
            temp_path = f"temp_vid_{hash(up.name)}.mp4"  # Unique filename
            with open(temp_path, "wb") as f: 
                f.write(up.read())
            
            cap = cv2.VideoCapture(temp_path)
            frame_holder = st.empty()
            
            # Get video info
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            progress_bar = st.progress(0)
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame = annotate(frame)
                frame_holder.image(frame, channels="BGR", use_column_width=True)
                
                # Update progress
                frame_count += 1
                progress_bar.progress(frame_count / total_frames)
                
                # Add small delay to make video playable
                if fps > 0:
                    st.time.sleep(1 / fps)
            
            cap.release()
            os.remove(temp_path)  # Cleanup
            progress_bar.empty()
            
        except Exception as e:
            st.error(f"Error processing video: {e}")

# ───────────────────────────── Footer Info ─────────────────────────────
st.markdown("---")
st.markdown("**Troubleshooting Tips:**")
st.markdown("- If webcam doesn't work, ensure camera permissions are enabled in your browser")
st.markdown("- For deployment issues, consider using platforms that support WebRTC (not Streamlit Cloud)")
st.markdown("- Model file 'best.pt' and 'roadsylabels.txt' must be in the same directory")