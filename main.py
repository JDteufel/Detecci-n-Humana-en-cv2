import cv2
import time
import os
from datetime import datetime
from filtros import procesar_baja_luz
from detector import detectar_persona

CARPETA_DATA = "data"
if not os.path.exists(CARPETA_DATA):
    os.makedirs(CARPETA_DATA)

cap = cv2.VideoCapture(0)
ultimo_tiempo = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar la cámara")
        break

    cv2.imshow("Camara en vivo", frame)

    # Cada 1 segundo procesar un frame
    if time.time() - ultimo_tiempo >= 1:
        ultimo_tiempo = time.time()

        # Procesar frame
        frame_editado = procesar_baja_luz(frame)

        # Detectar persona
        detectado, frame_resultado = detectar_persona(frame_editado)

        # Si hay detección → guardar imagen en TIFF sin pérdida
        if detectado:
            timestamp = datetime.now().strftime("%H_%M_%S_%d_%m_%Y")
            nombre_archivo = os.path.join(CARPETA_DATA, f"frame_{timestamp}.tiff")
            cv2.imwrite(nombre_archivo, frame_resultado, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
            print(f"[ALERTA] Persona detectada → Guardado: {nombre_archivo}")
        else:
            print("[INFO] No se detecta persona.")

        cv2.imshow("Procesada y detectada", frame_resultado)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
