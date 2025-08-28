import numpy as np
from PIL import Image
import streamlit as st
import cv2
import time
import os
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# ===================== Account System =====================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "users" not in st.session_state:
    st.session_state.users = {}

def account_system():
    if not st.session_state.authenticated:
        st.title("🔐 Login to ClearSky.AI")
        choice = st.radio("Choose an option", ["Login", "Create Account"])

        if choice == "Login":
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            if st.button("Login"):
                if username in st.session_state.users and st.session_state.users[username] == password:
                    st.session_state.authenticated = True
                    st.session_state.current_user = username
                    st.success("✅ Login successful!")
                else:
                    st.error("❌ Invalid username or password")
            st.stop()

        elif choice == "Create Account":
            new_user = st.text_input("Choose a username")
            new_pass = st.text_input("Choose a password", type="password")
            confirm_pass = st.text_input("Confirm password", type="password")
            st.markdown("### 📜 PDPA Terms & Conditions")
            st.write("""
            By creating an account, you agree that:
            - This demo app does not store or share your personal data permanently.
            - Uploaded images are processed only for haze detection and are not saved.
            - Any image containing faces will be blurred for privacy.
            - You can stop using this service at any time.
            """)
            agree = st.checkbox("✅ I have read and agree to the above terms.")

            if st.button("Create Account"):
                if not new_user or not new_pass:
                    st.error("❌ Username and password cannot be empty")
                elif new_pass != confirm_pass:
                    st.error("❌ Passwords do not match")
                elif new_user in st.session_state.users:
                    st.error("❌ Username already exists")
                elif not agree:
                    st.error("❌ You must agree to the PDPA terms to create an account")
                else:
                    st.session_state.users[new_user] = new_pass
                    st.success("✅ Account created! Please login now.")
            st.stop()

# ===================== Haze Detection =====================
def get_dark_channel(I, w=15):
    min_channel = np.min(I, axis=2)
    kernel = np.ones((w, w), np.uint8)
    dark_channel = cv2.erode(min_channel, kernel)
    return dark_channel

def analyze_dark_channel(dark_channel):
    mean_val = np.mean(dark_channel)
    if mean_val < 80:
        return "❌ No Haze Detected", mean_val
    elif 80 <= mean_val < 120:
        return "⚠️ Possible Light Haze", mean_val
    else:
        return "☁️ Heavy Haze Detected", mean_val

# ===================== Face Blurring with DNN =====================
# Use relative paths for Streamlit Cloud
face_net = cv2.dnn.readNetFromCaffe(
    "models/deploy.prototxt.txt",
    "models/res10_300x300_ssd_iter_140000.caffemodel"
)

def blur_faces_dnn(image):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    faces_detected = 0
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            face = image[y1:y2, x1:x2]
            blurred_face = cv2.GaussianBlur(face, (51, 51), 30)
            image[y1:y2, x1:x2] = blurred_face
            faces_detected += 1
    return image, faces_detected

# ===================== Video Transformer for WebRTC =====================
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.haze_values = []
        self.haze_label = "No data"
        self.haze_value = "No data"
        self.face_count = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Blur faces
        img, self.face_count = blur_faces_dnn(img)
        
        # Haze detection
        I_norm = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        I = np.asarray(I_norm, dtype=np.float64)
        dark = get_dark_channel(I, w=15)
        _, mean_val = analyze_dark_channel(dark)
        self.haze_values.append(mean_val)

        if len(self.haze_values) >= 5:
            avg_val = np.mean(self.haze_values)
            haze_result, _ = analyze_dark_channel(np.full_like(dark, avg_val))
            self.haze_label = haze_result
            self.haze_value = f"{avg_val:.2f}"
            self.haze_values.clear()
        
        # Add a text overlay to the video frame with the results
        overlay_text = f"Haze: {self.haze_label} | Dark Channel: {self.haze_value}"
        cv2.putText(img, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        if self.face_count > 0:
            face_text = f"Faces: {self.face_count} (Blurred)"
            cv2.putText(img, face_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ===================== Streamlit UI =====================
st.set_page_config(page_title="🌤 Haze Detection AI", layout="centered")
account_system()
st.title("🌤 Haze Detection AI")
st.markdown("Welcome to ClearSky.AI — Detect haze in real time or from images.")

# Sidebar menu
option = st.sidebar.radio("Choose Mode", ("📁 Upload Image", "📹 Real-Time Camera"))

if option == "📁 Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img)

        # Blur faces
        img_np, face_count = blur_faces_dnn(img_np)
        if face_count > 0:
            st.warning(f"⚠️ {face_count} face(s) detected and blurred for privacy.")

        with st.spinner("🔍 Analyzing image for haze..."):
            img_np_float = np.array(img_np, dtype=np.float64)
            img_norm = cv2.normalize(img_np_float, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            dark = get_dark_channel(img_norm, w=15)
            haze_result, mean_val = analyze_dark_channel(dark)

            st.image(img_np, caption="Uploaded Image (faces blurred)")
            st.markdown(f"### 🧪 Haze Detection: {haze_result}")
            st.write(f"📊 Avg. dark channel intensity: {mean_val:.2f}")

elif option == "📹 Real-Time Camera":
    st.markdown("### 📸 Real-Time Webcam Feed")
    webrtc_ctx = webrtc_streamer(
        key="haze-detection-webrtc",
        video_processor_factory=VideoTransformer,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
    )
    # The `webrtc_streamer` handles the whole process, no need for the old while loop.
    # The VideoTransformer class takes care of the frame processing.
