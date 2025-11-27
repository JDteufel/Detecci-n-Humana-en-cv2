import cv2
import mediapipe as mp
import math

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

def detectar_pose(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)
    return result.pose_landmarks  # Puede ser None

def calcular_angulo_head(landmarks):
    # landmark 2 = ojo izq / 5 = ojo der
    left = landmarks.landmark[2]
    right = landmarks.landmark[5]
    dx = right.x - left.x
    dy = right.y - left.y
    ang = math.degrees(math.atan2(dy, dx))
    return abs(ang)

def movimiento_cuerpo(landmarks_ref, landmarks_actual):
    hombro_ref = landmarks_ref.landmark[11]
    hombro_act = landmarks_actual.landmark[11]
    dx = abs(hombro_ref.x - hombro_act.x)
    dy = abs(hombro_ref.y - hombro_act.y)
    return (dx + dy) > 0.15

# SOLO compara posiciones actuales contra posición inicial
def comparar_pose(landmarks_ref, landmarks_actual):
    if landmarks_ref is None or landmarks_actual is None:
        return False
    return movimiento_cuerpo(landmarks_ref, landmarks_actual)
