import cv2
import time
import os
from filtros import procesar_baja_luz
from detector import detectar_persona

CARPETA_DATA = "data"
if not os.path.exists(CARPETA_DATA):
    os.makedirs(CARPETA_DATA)

cap = cv2.VideoCapture(0)
ultimo_tiempo = time.time()
contador = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar la cámara")
        break

    cv2.imshow("Camara en vivo", frame)

    # Cada 3 segundos procesar un frame
    if time.time() - ultimo_tiempo >= 3:
        ultimo_tiempo = time.time()
        # Guardar frame original
        nombre_archivo = os.path.join(CARPETA_DATA, f"frame_{contador}.jpg")
        cv2.imwrite(nombre_archivo, frame)
        print(f"[INFO] Imagen guardada: {nombre_archivo}")

        # Procesar frame
        frame_editado = procesar_baja_luz(frame)

        # Detectar persona
        detectado, frame_resultado = detectar_persona(frame_editado)
        if detectado:
            print("[INFO] Persona detectada!")
        else:
            print("[INFO] No se detecta persona.")

        cv2.imshow("Procesada y detectada", frame_resultado)
        contador += 1

    # Salir con ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
