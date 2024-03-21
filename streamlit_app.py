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
st.write('Choose detection method:')

# Choose detection method
option = st.selectbox('Select detection method', ('Image Upload', 'Webcam', 'Video Upload'))

if option == 'Image Upload':
    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Read the image
        frame = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
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
                        cv2.FONT_HERSHEY_COMPLEX, 0.75, (255, 0, 0), 1)

        # Display the frame with objects detected
        st.image(frame, channels="BGR", caption="Detected Objects", use_column_width=True)

elif option == 'Webcam':
    # Start webcam
    video_capture = cv2.VideoCapture(0)
    st.write('Press "q" to stop the webcam')
    #stop_button = st.empty()
    
    # Loop to capture frames and display the video with detections
    while True:
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
                        cv2.FONT_HERSHEY_COMPLEX, 0.75, (255, 0, 0), 1)

        # Display the frame with objects detected
        st.image(frame, channels="BGR", caption="Detected Objects", use_column_width=True)

        # Break the loop if "Stop Webcam" button is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release webcam
    video_capture.release()
    cv2.destroyAllWindows()

elif option == 'Video Upload':
    # Upload video through Streamlit
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])

    if uploaded_file is not None:
        # Save the uploaded video to a temporary file
        with open("temp_video.mp4", "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        # Open the temporary video file using VideoCapture
        video_capture = cv2.VideoCapture("temp_video.mp4")

        # Process each frame of the video
        while True:
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
                            cv2.FONT_HERSHEY_COMPLEX, 0.75, (255, 0, 0), 1)

            # Display the frame with objects detected
            st.image(frame, channels="BGR", caption="Detected Objects", use_column_width=True)

        # Release resources
        video_capture.release()
        cv2.destroyAllWindows()
