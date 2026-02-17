"""
Face Landmarker - Detects and draws facial landmarks on images
"""
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles
import numpy as np
import cv2

MODEL_PATH = "/Users/nikhiljain/Desktop/Coding Projects/face-recognition-ai/Face Landmarker Task.task"

# Create detector (loaded once)
_detector = None

def get_detector():
    global _detector
    if _detector is None:
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
            output_face_blendshapes=True,
            num_faces=1
        )
        _detector = vision.FaceLandmarker.create_from_options(options)
    return _detector


def process_image(image_bytes: bytes) -> bytes:
    """
    Takes image bytes, returns image bytes with face landmarks drawn.
    
    Input: Raw image bytes (from uploaded file)
    Output: JPEG bytes with landmarks drawn on face
    """
    # 1. Decode bytes to image
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not read image")
    
    # 2. Convert BGR -> RGB for MediaPipe
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    # 3. Detect face landmarks
    result = get_detector().detect(mp_image)
    if not result.face_landmarks:
        raise ValueError("No face detected")
    
    # 4. Draw landmarks on image
    annotated = np.copy(rgb)
    for face in result.face_landmarks:
        # Draw mesh
        drawing_utils.draw_landmarks(
            image=annotated,
            landmark_list=face,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style()
        )
        # Draw contours (eyes, lips, eyebrows)
        drawing_utils.draw_landmarks(
            image=annotated,
            landmark_list=face,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style()
        )
    
    # 5. Convert back to BGR and encode as JPEG
    bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    return buffer.tobytes()


# Test when run directly
if __name__ == "__main__":
    with open("test_image.jpg", "rb") as f:
        result = process_image(f.read())
    with open("result.jpg", "wb") as f:
        f.write(result)
    print("Saved result.jpg")
