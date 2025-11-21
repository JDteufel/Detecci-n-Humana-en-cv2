import mediapipe as mp
import math

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

def detectar_pose(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return pose.process(rgb).pose_landmarks

def calcular_angulo_head(landmarks):
    left = landmarks.landmark[2]   # LEFT_EYE
    right = landmarks.landmark[5]  # RIGHT_EYE
    dx = right.x - left.x
    dy = right.y - left.y
    ang = math.degrees(math.atan2(dy, dx))
    return abs(ang)

def movimiento_cuerpo(landmarks_ref, landmarks_actual):
    hombro_ref = landmarks_ref.landmark[11]
    hombro_act = landmarks_actual.landmark[11]
    dx = abs(hombro_ref.x - hombro_act.x)
    dy = abs(hombro_ref.y - hombro_act.y)
    return (dx + dy) > 0.15  # Movimiento exagerado = alerta

def comparar_pose(landmarks_ref, landmarks_actual):
    if landmarks_ref is None or landmarks_actual is None:
        return False  # Si no hay cuerpo en alguno, lo tratamos normal

    # 1. Cabeza giró mucho → alarma
    ang_ref = calcular_angulo_head(landmarks_ref)
    ang_act = calcular_angulo_head(landmarks_actual)
    if abs(ang_act - ang_ref) > 30:
        return True

    # 2. Cuerpo se movió fuerte → alarma
    if movimiento_cuerpo(landmarks_ref, landmarks_actual):
        return True

    return False
