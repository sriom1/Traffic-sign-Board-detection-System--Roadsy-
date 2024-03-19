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
st.title('ROADSY:Automated Traffic Sign Detection and Classification')
st.write('Upload an image and the model will detect traffic signs')

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Read the image
    frame = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Resize the frame to a fixed size (if necessary)
    frame = cv2.resize(frame, (1020, 500))

    # Perform object detection on the frame
    results = model.predict(frame)
    detections = results[0].boxes.data
    px = pd.DataFrame(detections).astype("float")

    # Display the detected objects
    st.image(frame, channels="BGR", caption="Uploaded Image with Object Detection", use_column_width=True)

    # Display the detected objects and their class labels
    tumour_count = 0
    for index, row in px.iterrows():
        x1, y1, x2, y2, _, d = map(int, row)
        c = class_list[d]

        # Draw bounding boxes and class labels on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0, 255), 2)
        cv2.putText(frame, str(c), (x1, y1),
                    cv2.FONT_HERSHEY_COMPLEX, 1.25, (0,255,0), 1)

        # Check if the detected object is a tumour
        if c.lower() == "tumour":
            tumour_count += 1

    # Display the frame with objects detected
    st.image(frame, channels="BGR", caption="Detected Objects", use_column_width=True)
    # Use HTML and CSS to style the text
    st.markdown(f"""
    <style>
        .tumour_count {{
            font-size: 24px;
            color: red;
        }}
    </style>
    
    """, unsafe_allow_html=True)
