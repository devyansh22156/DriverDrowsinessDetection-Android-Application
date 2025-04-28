import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import math
from flask import Flask, render_template, Response, jsonify, request
import os

# Define your CNN+LSTM model (must match the architecture used during training)
class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        # 1D CNN Layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        # LSTM Layer
        self.lstm = nn.LSTM(input_size=16, hidden_size=32, num_layers=1, batch_first=True)
        # Fully Connected Layer for 2 classes: Drowsy / Non-Drowsy
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        # x shape: (batch, 1, features)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # Permute to (batch, sequence_length, input_size) for LSTM
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        # Use the output from the last LSTM timestep
        x = x[:, -1, :]
        x = self.fc(x)
        return x


def load_drowsiness_model(model_path="best_model.pth"):  # adjust path if needed
    model = CNN_LSTM()
    state_dict = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Initialize MediaPipe Face Mesh with lower thresholds
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

app = Flask(__name__)

# Global variable to store latest computed features
latest_features = {
    "ear": None,
    "mar": None,
    "head_pose": None,
    "label": None,
    "confidence": None
}

# Load the trained drowsiness model
model = load_drowsiness_model("best_model.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def calculate_ear(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)


def calculate_mar(mouth):
    A = np.linalg.norm(mouth[1] - mouth[7])
    B = np.linalg.norm(mouth[2] - mouth[6])
    C = np.linalg.norm(mouth[0] - mouth[4])
    return (A + B) / (2.0 * C)


def estimate_head_pose_mediapipe(landmarks_np):
    left_eye_outer = landmarks_np[33]
    right_eye_outer = landmarks_np[263]
    dx = right_eye_outer[0] - left_eye_outer[0]
    dy = right_eye_outer[1] - left_eye_outer[1]
    return math.degrees(math.atan2(dy, dx))


def extract_features_from_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    h, w, _ = frame.shape
    landmarks_np = np.array([[int(lm.x * w), int(lm.y * h)] for lm in landmarks])

    left_eye_indices = [33, 160, 158, 133, 153, 144]
    right_eye_indices = [263, 387, 385, 362, 380, 373]
    mouth_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]

    try:
        left_eye = landmarks_np[left_eye_indices]
        right_eye = landmarks_np[right_eye_indices]
        mouth = landmarks_np[mouth_indices]
    except IndexError:
        return None

    ear_left = calculate_ear(left_eye)
    ear_right = calculate_ear(right_eye)
    ear_avg = (ear_left + ear_right) / 2.0
    mar = calculate_mar(mouth)
    head_pose = estimate_head_pose_mediapipe(landmarks_np)
    return np.array([ear_avg, mar, head_pose], dtype=np.float32)


def gen_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not accessible")
        return
    while True:
        success, frame = cap.read()
        if not success:
            break

        features = extract_features_from_frame(frame)
        if features is not None:
            features_tensor = torch.tensor(features.reshape(1, 1, -1), dtype=torch.float32).to(device)
            with torch.no_grad():
                outputs = model(features_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                conf = probabilities[0][predicted_class].item()
            label = "Drowsy" if predicted_class == 1 else "Non Drowsy"
            color = (0, 0, 255) if predicted_class == 1 else (0, 255, 0)
            cv2.putText(frame, f"{label} ({conf*100:.1f}%)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            latest_features.update({
                "ear": float(features[0]),
                "mar": float(features[1]),
                "head_pose": float(features[2]),
                "label": label,
                "confidence": conf * 100.0
            })
        else:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()
import os
from datetime import datetime

RECEIVED_FOLDER = 'received_images'
os.makedirs(RECEIVED_FOLDER, exist_ok=True)

@app.route('/predict', methods=['POST'])
def predict_from_android():
    if 'image' not in request.files:
        return jsonify({'label': 'No image uploaded', 'confidence': 0.0}), 400

    image_bytes = request.files['image'].read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({'label': 'Invalid frame', 'confidence': 0.0}), 400

    print(f"[predict] received frame of shape {frame.shape}")

    # --- Save image to folder if less than 5 exist ---
    existing_images = sorted(os.listdir(RECEIVED_FOLDER))
    if len(existing_images) < 5:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"frame_{timestamp}.jpg"
        filepath = os.path.join(RECEIVED_FOLDER, filename)
        cv2.imwrite(filepath, frame)
        print(f"[predict] Saved image as {filename}")

    features = extract_features_from_frame(frame)
    if features is None:
        return jsonify({'label': 'No face detected', 'confidence': 0.0}), 200

    features_tensor = torch.tensor(features.reshape(1, 1, -1), dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(features_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        idx = int(torch.argmax(probabilities, dim=1))
        conf = float(probabilities[0][idx].item())

    label = "Drowsy" if idx == 1 else "Non Drowsy"
    print(label)
    return jsonify({
        'label': label,
        'confidence': round(conf * 100, 2),
        'ear': float(features[0]),
        'mar': float(features[1]),
        'head_pose': float(features[2])
    }), 200



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/features')
def features():
    return jsonify(latest_features)


if __name__ == '__main__':
    app.run(debug=True, host="192.168.29.140", port=5000)
