import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def analyze_posture(frame):

    image = frame.copy()
    h, w, _ = image.shape

    # convert BGR -> RGB without OpenCV
    image_rgb = image[:, :, ::-1]

    results = pose.process(image_rgb)

    posture_text = "Detecting..."
    color = (255, 255, 255)
    angle = 0

    if results.pose_landmarks:

        landmarks = results.pose_landmarks.landmark

        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]

        left_shoulder_x = int(left_shoulder.x * w)
        left_shoulder_y = int(left_shoulder.y * h)

        right_shoulder_x = int(right_shoulder.x * w)
        right_shoulder_y = int(right_shoulder.y * h)

        left_hip_x = int(left_hip.x * w)
        left_hip_y = int(left_hip.y * h)

        right_hip_x = int(right_hip.x * w)
        right_hip_y = int(right_hip.y * h)

        shoulder_mid_x = int((left_shoulder_x + right_shoulder_x) / 2)
        shoulder_mid_y = int((left_shoulder_y + right_shoulder_y) / 2)

        hip_mid_x = int((left_hip_x + right_hip_x) / 2)
        hip_mid_y = int((left_hip_y + right_hip_y) / 2)

        spine_vector = np.array([
            shoulder_mid_x - hip_mid_x,
            shoulder_mid_y - hip_mid_y
        ])

        vertical_vector = np.array([0, -1])

        cos_theta = np.dot(spine_vector, vertical_vector) / (
            np.linalg.norm(spine_vector) * np.linalg.norm(vertical_vector)
        )

        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_theta))

        vertical_diff = abs(shoulder_mid_x - hip_mid_x)

        if angle < 10 and vertical_diff < 20:
            posture_text = "Good Posture"
            color = (0,255,0)

        elif angle < 20:
            posture_text = "Slight Bend"
            color = (0,255,255)

        else:
            posture_text = "Bad Posture"
            color = (0,0,255)

    return posture_text, color, angle, results


import streamlit as st
import cv2
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import mediapipe as mp
import pyttsx3
import time
import threading
from queue import Queue

# -------------------- VOICE SYSTEM  --------------------

engine = pyttsx3.init()
alert_queue = Queue()
speech_lock = threading.Lock()

def voice_worker():
    while True:
        message = alert_queue.get()
        if message:
            with speech_lock:   # prevents crash
                try:
                    engine.say(message)
                    engine.runAndWait()
                except:
                    pass  # prevent app crash


threading.Thread(target=voice_worker, daemon=True).start()

# -------------------- ALERT VARIABLES --------------------


alert_interval = 5
trigger_time = 3

# -------------------- MEDIAPIPE --------------------

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# -------------------- STREAMLIT UI --------------------

st.title("PostureSense")
st.write("Real-time posture monitoring system")

enable_voice = st.checkbox("Enable Voice Alert", value=True)

# -------------------- VIDEO PROCESSOR --------------------

class PostureProcessor(VideoProcessorBase):

    def __init__(self):
        self.last_alert_time = 0
        self.bad_posture_start_time = None

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        posture_text, color, angle, results = analyze_posture(img)

        current_time = time.time()

        # -------------------- SMART ALERT LOGIC --------------------

        if enable_voice:

            if posture_text == "Bad Posture":

                # Start timer
                if self.bad_posture_start_time is None:
                    self.bad_posture_start_time = current_time

                # Trigger only after continuous bad posture
                if current_time - self.bad_posture_start_time > trigger_time:

                    # Repeat every interval
                    if current_time - self.last_alert_time > alert_interval:

                        alert_queue.put("Please correct your posture")

                        self.last_alert_time = current_time

            else:
                # Reset if posture improves
                self.bad_posture_start_time = None

        # -------------------- DRAW SKELETON --------------------

        if results.pose_landmarks:

            mp_drawing.draw_landmarks(
                img,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=color, thickness=3, circle_radius=4),
                mp_drawing.DrawingSpec(color=color, thickness=3)
            )

        # -------------------- TEXT DISPLAY --------------------

        cv2.putText(img,
                    posture_text,
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2)

        cv2.putText(img,
                    f"Angle: {int(angle)}",
                    (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# -------------------- START STREAM --------------------

webrtc_streamer(
    key="posture",
    video_processor_factory=PostureProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

































































