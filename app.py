import cv2
import time
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

from g_helper import bgr2rgb, rgb2bgr, mirrorImage
from fp_helper import pipelineHeadTiltPose, draw_face_landmarks_fp
from ms_helper import pipelineMouthState
from es_helper import pipelineEyesState

# Open video file or capture from webcam
video_path = "/kaggle/input/situations/sit1.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video file or webcam.")
    exit()

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Output video writer
out = cv2.VideoWriter('/kaggle/working/sit1.mp4', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

start_time = time.time()

with mp_face_mesh.FaceMesh(max_num_faces=2, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()

        # Mirror image (Optional)
        image = mirrorImage(image)

        # Generate face mesh
        results = face_mesh.process(bgr2rgb(image))

        # Processing Face Landmarks
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # FACE MESH ----------------------------------------
                draw_face_landmarks_fp(image, face_landmarks)

                # HEAD TILT POSE -----------------------------------
                head_tilt_pose = pipelineHeadTiltPose(image, face_landmarks)

                # MOUTH STATE --------------------------------------
                mouth_state = pipelineMouthState(image, face_landmarks)

                # EYES STATE ---------------------------------------
                r_eyes_state, l_eyes_state = pipelineEyesState(image, face_landmarks)

        # Write frame to output video
        out.write(image)
    cap.release()
    out.release()

    end_time = time.time()
    # Calculate execution time
    execution_time = end_time - start_time
    print(f"Execution Time: {execution_time:.2f} seconds")
