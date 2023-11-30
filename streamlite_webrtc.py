import av
import cv2
import streamlit as st
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode


# Inicializar el modelo de manos de Mediapipe
mp_hands = mp.solutions.hands
hands_model = mp_hands.Hands()

def live_hand_detection(play_state):

    class HandProcessor(VideoProcessorBase):

        def __init__(self) -> None:
            self.results = None

        def detect_hand(self, image):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.results = hands_model.process(image_rgb)  # Usar el modelo de manos inicializado
            if self.results.multi_hand_landmarks:
                for hand_landmarks in self.results.multi_hand_landmarks:
                    h, w, _ = image.shape
                    landmarks = [(int(l.x * w), int(l.y * h)) for l in hand_landmarks.landmark]
                    cv2.rectangle(image, (min(landmarks)[0], min(landmarks, key=lambda x: x[1])[1]),
                                  (max(landmarks)[0], max(landmarks, key=lambda x: x[1])[1]),
                                  (0, 255, 0), 2)
            return image

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="bgr24")
            hand_detected_image = self.detect_hand(image)

            return av.VideoFrame.from_ndarray(hand_detected_image, format="bgr24")

    stream = webrtc_streamer(
        key="hand-detection",
        mode=WebRtcMode.SENDRECV,
        desired_playing_state=play_state,
        video_processor_factory=HandProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

play_state = True
live_hand_detection(play_state)
