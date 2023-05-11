import cv2
import streamlit as st
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, ClientSettings

st.set_page_config(page_title='Face Blur App', page_icon=':smiley:')

st.title('Face Blur App')

# Load the DNN model.
modelFile = 'res10_300x300_ssd_iter_140000.caffemodel'
configFile = 'deploy.prototxt'

# Read the model and create a network object.
net = cv2.dnn.readNetFromCaffe(prototxt=configFile, caffeModel=modelFile)

def blur_ellipse(face, factor=1.5):
    h, w  = face.shape[:2]
    mask = np.zeros((h, w), np.uint8)

    # Create an elliptical mask.
    cv2.ellipse(mask, (w//2, h//2), (int(w/factor), int(h/factor)), 0, 0, 360, 255, -1)

    # Apply the mask and blur the face.
    blurred = cv2.GaussianBlur(face, (99, 99), 30)
    face = np.where(mask[..., None] == 255, blurred, face)

    return face

class FaceBlurVideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = face_blur_ellipse(img, net)
        out_frame = av.VideoFrame.from_ndarray(img, format="bgr24")
        return out_frame

def face_blur_ellipse(image, net, factor=1.5, detection_threshold=0.9):
    img = image.copy()

    # Convert the image into a blob format.
    blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(300,300), mean=[104, 117, 123])

    # Pass the blob to the DNN model.
    net.setInput(blob)

    # Retrieve detections from the DNN model.
    detections = net.forward()

    (h, w) = img.shape[:2]

    # Process the detections.
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > detection_threshold:

            # Extract the bounding box coordinates from the detection.
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # Extract the face ROI.
            face = img[y1:y2, x1:x2]

            face = blur_ellipse(face, factor=factor)

            # Replace the detected face with the blurred one.
            img[y1:y2, x1:x2] = face

    return img

webrtc_ctx = webrtc_streamer(
    key="face-blur",
    video_processor_factory=FaceBlurVideoProcessor,
    client_settings=ClientSettings(
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    ),
)

if not webrtc_ctx.state.playing:
    st.write("Click the 'start' button below to open your webcam and apply the face blur.")
else:
    st.write("Your face is being blurred!")

