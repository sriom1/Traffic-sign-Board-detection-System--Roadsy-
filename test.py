import cv2
import pandas as pd
import streamlit as st
from ultralytics import YOLO
import numpy as np

# Load the pre-trained YOLO model
model = YOLO("best.pt")

# Read the COCO class list from a file
with open("roadsylabels.txt", "r") as my_file:
    class_list = my_file.read().split("\n")

# Streamlit interface
st.title('ROADSY: Automated Traffic Sign Detection and Classification')
st.write('Press the button to start webcam detection')

# Create a placeholder for the stop button
stop_button_placeholder = st.empty()

# Start webcam
clicked = st.button("Start Webcam")
if clicked:
    # Start webcam capture
    video_capture = cv2.VideoCapture(0)

    # Create a function to read and process frames
    def process_frames(video_capture):
        # Create a variable to keep track of whether to stop or continue processing frames
        stop_flag = False

        while not stop_flag:
            ret, frame = video_capture.read()
            if not ret:
                break

            # Perform object detection on the frame
            results = model.predict(frame)
            detections = results[0].boxes.data
            px = pd.DataFrame(detections).astype("float")

            # Display the detected objects and their class labels
            for index, row in px.iterrows():
                x1, y1, x2, y2, _, d = map(int, row)
                c = class_list[d]

                # Draw bounding boxes and class labels on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, str(c), (x1, y1),
                            cv2.FONT_HERSHEY_COMPLEX, 1.25, (0, 255, 0), 1)

            # Convert the frame to RGB for displaying in Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Display the frame with objects detected
            st.image(frame_rgb, channels="RGB", caption="Detected Objects", use_column_width=True)

            # Check if "Stop Webcam" button is clicked
            if stop_button_placeholder.button("Stop Webcam"):
                stop_flag = True
                break

    # Call the function to start processing frames
    process_frames(video_capture)

    # Release webcam
    video_capture.release()
