# Roadsy: Real-time Traffic Sign Board Detection System

This project is a real-time traffic sign board detection system built using [YOLOv8](https://github.com/ultralytics/ultralytics) for object detection and [Streamlit](https://streamlit.io/) for the web interface.

## Features

- Detects traffic sign boards in images and video streams
- Real-time detection using YOLOv8
- Simple and interactive web interface with Streamlit
- Supports webcam and image uploads
- Displays detection results with bounding boxes and labels

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/roadsy.git
   cd roadsy/Traffic-sign-Board-detection-System--Roadsy-
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```sh
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Streamlit app:**
   ```sh
   streamlit run app.py
   ```

2. **Open the web interface:**
   - The app will open in your browser at `http://localhost:8501`
   - Upload an image or use your webcam to start detecting traffic signs

## Project Structure

```
Traffic-sign-Board-detection-System--Roadsy-/
│
├── streamlit_app.py                 # Main Streamlit application
├── yolov8_model.py        # YOLOv8 model loading and inference
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── data/                  # Sample images and videos
└── utils/                 # Utility functions
```

## Model

- The project uses YOLOv8 for traffic sign detection.
- You can use a pre-trained model or train your own on a custom dataset.
- Place your YOLOv8 weights file in the project directory and update the path in `yolov8_model.py`.

## Customization

- To add new traffic sign classes, retrain YOLOv8 with your dataset.
- Modify `streamlit_app.py ` to adjust the Streamlit interface as needed.

## Requirements

- Python 3.8+
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Streamlit
- OpenCV
- Other dependencies listed in `requirements.txt`

## License

This project is licensed under the MIT License.

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Streamlit](https://streamlit.io/)

---

Feel free to contribute or open issues for improvements and bug fixes.
