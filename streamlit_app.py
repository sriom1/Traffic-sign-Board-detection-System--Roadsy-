# ROADSY ‑ Streamlit app with Image, Video and **Webcam** support
# ---------------------------------------------------------------
# Requirements (run once):
#   pip install streamlit streamlit‑webrtc ultralytics opencv-python-headless numpy pandas

import cv2
import pandas as pd
import streamlit as st
from ultralytics import YOLO
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# ───────────────────────── UI  heading & background ─────────────────────────
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

# ─────────────────────────  load model & labels  ────────────────────────────
model = YOLO("best.pt")                       # your trained weights
with open("roadsylabels.txt") as f:
    class_list = f.read().splitlines()

# ─────────────────────────── helper for drawing  ────────────────────────────
def annotate(frame):
    """run YOLO and draw boxes; returns BGR image"""
    results = model.predict(frame, verbose=False)
    if not results:
        return frame
    boxes = results[0].boxes.data.cpu().numpy()  # x1,y1,x2,y2,conf,class
    for x1, y1, x2, y2, *_ , cls in boxes:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
        cv2.putText(
            frame,
            class_list[int(cls)],
            (int(x1), int(y1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255,0,0),
            2,
            cv2.LINE_AA,
        )
    return frame

# ──────────────────────── streamlit-webrtc class  ───────────────────────────
class SignDetector(VideoTransformerBase):
    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = annotate(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ───────────────────────────── main selector  ───────────────────────────────
option = st.selectbox("Select detection method", ("Image Upload", "Webcam", "Video Upload"))

# 1️⃣ IMAGE   ---------------------------------------------------------------
if option == "Image Upload":
    up = st.file_uploader("Choose an image…", type=["jpg","jpeg","png"])
    if up:
        img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
        img = annotate(img)
        st.image(img, channels="BGR", caption="Detected Objects", use_column_width=True)

# 2️⃣ WEBCAM  ---------------------------------------------------------------
elif option == "Webcam":
    st.write("Allow camera access; stream will appear below.")
    webrtc_streamer(key="roadsy-webcam", video_transformer_factory=SignDetector)

# 3️⃣ VIDEO FILE  -----------------------------------------------------------
elif option == "Video Upload":
    up = st.file_uploader("Choose a video…", type=["mp4","mov","avi","mkv"])
    if up:
        # save to temp file
        with open("temp_vid.mp4","wb") as f: f.write(up.read())
        cap = cv2.VideoCapture("temp_vid.mp4")
        frame_holder = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = annotate(frame)
            frame_holder.image(frame, channels="BGR", use_column_width=True)
        cap.release()
