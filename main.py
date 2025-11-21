import cv2
import time
import os
import math
from datetime import datetime
from detector import detectar_pose, comparar_pose

CARPETA_DATA = "data"
if not os.path.exists(CARPETA_DATA):
    os.makedirs(CARPETA_DATA)

cap = cv2.VideoCapture(0)
t0 = time.time()
pose_inicial = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar la cámara")
        break

    pose_actual, cantidad = detectar_pose(frame)

    # ----------- MOSTRAR EN PANTALLA -----------
    info = f"Personas: {cantidad}"
    cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)

    if pose_actual:
        # Ángulo de cabeza (landmarks ojo izquierdo y ojo derecho)
        ojo_izq = pose_actual[2]     # landmark 2
        ojo_der = pose_actual[5]     # landmark 5
        dx = ojo_der.x - ojo_izq.x
        dy = ojo_der.y - ojo_izq.y
        angulo = math.degrees(math.atan2(dy, dx))

        cv2.putText(frame, f"Angulo cabeza: {angulo:.1f}°",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    else:
        angulo = None

    # ----------- GUARDAR POSE INICIAL EN SEG 5 -----------
    if pose_inicial is None and time.time() - t0 >= 5:
        if pose_actual:
            pose_inicial = pose_actual.copy()
            print("[INFO] Pose inicial guardada (segundo 5).")

    # ----------- MULTIPERSONA (¡ALARMA!) -----------
    if cantidad > 1:
        timestamp = datetime.now().strftime("%H_%M_%S_%d_%m_%Y")
        nombre = os.path.join(CARPETA_DATA, f"MULTIPERSONA_{timestamp}.jpg")
        cv2.imwrite(nombre, frame)
        print(f"[ALERTA] MÁS DE UNA PERSONA → Guardado: {nombre}")
        continue

    # ----------- COMPARACIÓN DE POSES -----------
    if pose_inicial and pose_actual:
        cambio = comparar_pose(pose_inicial, pose_actual)

        cv2.putText(frame, f"Desplazamiento: {cambio:.3f}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)

        if cambio > 0.08 or abs(angulo) > 30:
            timestamp = datetime.now().strftime("%H_%M_%S_%d_%m_%Y")
            nombre = os.path.join(CARPETA_DATA, f"ALERTA_{timestamp}.jpg")
            cv2.imwrite(nombre, frame)
            print(f"[ALERTA] Movimiento MUY brusco → Guardado: {nombre}")

    cv2.imshow("EXAMEN EN CURSO", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
