"""
AI 后端 — MediaPipe Pose Landmarker
"""
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class PoseLandmarkerBackend:
    """MediaPipe 姿态估计后端"""
    
    def __init__(self, model_path: str = "", min_confidence: float = 0.5):
        if not model_path:
            model_path = os.path.expanduser("~/.hermes/cache/pose_landmarker_lite.task")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"模型文件不存在: {model_path}\n"
                f"下载: curl -L -o {model_path} "
                f"https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
            )
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=min_confidence,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._detector = vision.PoseLandmarker.create_from_options(options)
        self._landmark_names = [
            "nose", "left_eye_inner", "left_eye", "left_eye_outer",
            "right_eye_inner", "right_eye", "right_eye_outer",
            "left_ear", "right_ear", "mouth_left", "mouth_right",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_pinky", "right_pinky",
            "left_index", "right_index", "left_thumb", "right_thumb",
            "left_hip", "right_hip", "left_knee", "right_knee",
            "left_ankle", "right_ankle", "left_heel", "right_heel",
            "left_foot_index", "right_foot_index"
        ]
    
    def detect(self, frame) -> dict:
        """检测一帧的姿态关键点"""
        import cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_img)
        
        if not result.pose_landmarks:
            return None
        
        landmarks = result.pose_landmarks[0]
        return {
            self._landmark_names[i]: {
                "x": lm.x, "y": lm.y, "z": lm.z,
                "visibility": lm.visibility,
                "presence": lm.presence,
            }
            for i, lm in enumerate(landmarks)
        }
    
    def close(self):
        self._detector.close()
