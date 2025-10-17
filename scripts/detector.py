import cv2
import numpy as np
from pathlib import Path

# --- Carpetas absolutas ---
FILTRADA_DIR = Path(r"C:\Users\jgome\Desktop\ArchivosUni\Procesamiento de Imágenes\Detecci-n-Humana-en-cv2\data\filtrada")
DETECTADA_DIR = Path(r"C:\Users\jgome\Desktop\ArchivosUni\Procesamiento de Imágenes\Detecci-n-Humana-en-cv2\data\detectada")

DETECTADA_DIR.mkdir(parents=True, exist_ok=True)

# --- Nombres de archivos ---
IMG_FILTRADA = FILTRADA_DIR / "frame_preprocesado.jpg"
IMG_DETECTADA = DETECTADA_DIR / "frame_detectado.jpg"

# --- Leer imagen preprocesada ---
try:
    with open(str(IMG_FILTRADA), 'rb') as f:
        file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        print("Error: La imagen preprocesada no se pudo cargar")
        exit()
    else:
        print(f"Imagen preprocesada cargada correctamente desde {IMG_FILTRADA}")
except Exception as e:
    print("Error al leer la imagen preprocesada:", e)
    exit()

# --- Inicializar HOG Descriptor para detectar personas ---
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# --- Detectar humanos ---
# scale=1.05 es recomendable para detección en imágenes pequeñas; winStride=(4,4) y padding=(8,8)
boxes, weights = hog.detectMultiScale(img, winStride=(4, 4), padding=(8, 8), scale=1.05)

# Dibujar rectángulos sobre las detecciones
for i, (x, y, w, h) in enumerate(boxes):
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, f'person {i+1}', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

# --- Guardar imagen detectada ---
try:
    with open(str(IMG_DETECTADA), 'wb') as f:
        _, buf = cv2.imencode('.jpg', img)
        f.write(buf)
    print(f"Imagen con detecciones guardada correctamente en: {IMG_DETECTADA}")
except Exception as e:
    print("Error al guardar la imagen detectada:", e)
    exit()

# --- Señal de humano detectado ---
if len(boxes) > 0:
    print("Humano detectado: ¡Activar iluminación!")
else:
    print("No se detectaron humanos.")
