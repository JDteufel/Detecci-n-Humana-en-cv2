import cv2
import time
import os
import math
from datetime import datetime
from collections import deque
from detector import detectar_pose, calcular_angulo_head, comparar_pose

CARPETA_DATA = "data"
if not os.path.exists(CARPETA_DATA):
    os.makedirs(CARPETA_DATA)

cap = cv2.VideoCapture(0)
t0 = time.time()
pose_inicial = None

# Historial de movimientos y ángulos de cabeza
hist_angulos = deque(maxlen=30)   # ~1 segundo (30 FPS)
hist_movs = deque(maxlen=30)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar la cámara")
        break

    landmarks = detectar_pose(frame)

    # Mostrar cantidad de personas (solo detectamos 1 o 0)
    cantidad = 1 if landmarks else 0
    cv2.putText(frame, f"Personas: {cantidad}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Guardar pose inicial en el segundo 5
    if pose_inicial is None and time.time() - t0 >= 5:
        if landmarks:
            pose_inicial = landmarks
            print("[INFO] Pose inicial guardada (segundo 5).")

    # Si hay pose → analizar
    if landmarks:
        angulo = calcular_angulo_head(landmarks)
        cv2.putText(frame, f"Ángulo cabeza: {angulo:.1f}°", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        # Comparar movimiento contra pose inicial
        mov_actual = comparar_pose(pose_inicial, landmarks) if pose_inicial else False

        # Guardar en historial
        hist_angulos.append(abs(angulo))
        hist_movs.append(1 if mov_actual else 0)

    else:
        hist_angulos.append(0)
        hist_movs.append(0)
        angulo = 0

    # Mostramos promedio cuando ya haya historial completo
    if len(hist_angulos) == 30:
        ang_prom = sum(hist_angulos) / len(hist_angulos)
        mov_prom = sum(hist_movs) / len(hist_movs)

        cv2.putText(frame, f"Prom angulo: {ang_prom:.1f}°", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        cv2.putText(frame, f"Prom mov: {mov_prom:.2f}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)

        # ALERTA → movimiento sostenido
        if ang_prom > 40 or mov_prom > 0.5:
            ts = datetime.now().strftime("%H_%M_%S_%d_%m_%Y")
            nombre = os.path.join(CARPETA_DATA, f"SOSPECHA_LARGA_{ts}.jpg")
            cv2.imwrite(nombre, frame)
            print(f"[ALERTA] Movimiento largo sospechoso → Guardado: {nombre}")

    cv2.imshow("EXAMEN EN CURSO", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
