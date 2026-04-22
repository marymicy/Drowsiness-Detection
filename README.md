#  Real-Time Driver Drowsiness Detection System

**Drowsiness Detection** is a safety-critical computer vision application designed to prevent road accidents by monitoring driver alertness in real-time. The system utilizes facial landmark detection to calculate eye closure duration and triggers an audible alarm when fatigue is detected.

##  Key Features
* **Real-Time Monitoring**: Processes live video feed with minimal latency.
* **EAR Logic**: Uses Eye Aspect Ratio (EAR) for precise blink and sleep detection.
* **Facial Landmark Mapping**: Leverages a 68-point predictor for robust face tracking.
* **Instant Alerts**: Integrated audio alarm system that triggers after a specific frame threshold.
* **Low Hardware Demand**: Optimized to run efficiently on standard CPU architectures.

##  The Science: Eye Aspect Ratio (EAR)
The system localizes the eyes and identifies 6 vertices for each eye. The **Eye Aspect Ratio** is calculated using the Euclidean distance between these points:

$$EAR = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2||p_1 - p_4||}$$

* **Open Eyes**: EAR remains relatively constant (approx. 0.3).
* **Closed Eyes**: EAR rapidly approaches zero.
* **Threshold**: If EAR falls below **0.25** for more than **48 consecutive frames** (approx. 2 seconds at standard frame rates), an alert is triggered.

## 🛠️ Tech Stack
* **Language**: Python 3.x
* **Libraries**: OpenCV, Dlib, Imutils, Scipy, Pygame (for audio)
* **Pre-trained Model**: `shape_predictor_68_face_landmarks.dat`

##  Project Structure
* `driver_drowsiness.py`: Main application script.
* `shape_predictor_68_face_landmarks.dat`: The Dlib shape predictor model.
* `requirements.txt`: List of necessary Python dependencies.
* `/assets`: Store demonstration screenshots or alarm sound files (`alarm.wav`) here.

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.x installed. You will also need a working webcam.

### 2. Installation
Clone the repository and install the required dependencies:
```bash
git clone [https://github.com/marymicy/Drowsiness-Detection.git](https://github.com/marymicy/Drowsiness-Detection.git)
cd Drowsiness-Detection
pip install -r requirements.txt


