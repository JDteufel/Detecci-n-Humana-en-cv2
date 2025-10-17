import cv2
import numpy as np
from pathlib import Path

# --- Carpetas absolutas ---
ORIGINAL_DIR = Path(r"C:\Users\jgome\Desktop\ArchivosUni\Procesamiento de Imágenes\Detecci-n-Humana-en-cv2\data\original")
FILTRADA_DIR = Path(r"C:\Users\jgome\Desktop\ArchivosUni\Procesamiento de Imágenes\Detecci-n-Humana-en-cv2\data\filtrada")
FILTRADA_DIR.mkdir(parents=True, exist_ok=True)

# --- Archivo de entrada y salida ---
IMG_ORIGINAL = ORIGINAL_DIR / "frame_original.jpg"
IMG_FILTRADA = FILTRADA_DIR / "frame_preprocesado.jpg"

# --- Leer imagen original ---
try:
    with open(str(IMG_ORIGINAL), 'rb') as f:
        file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        print("Error: No se pudo cargar la imagen original")
        exit()
except Exception as e:
    print("Error al leer la imagen original:", e)
    exit()

print(f"Imagen original cargada correctamente desde {IMG_ORIGINAL}")

# --- Convertir a gris ---
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- Equalización adaptativa (CLAHE) ---
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced = clahe.apply(gray)

# --- Suavizado ligero para reducir ruido ---
blurred = cv2.GaussianBlur(enhanced, (3,3), 0)

# --- Detección de bordes Canny ---
edges = cv2.Canny(blurred, threshold1=30, threshold2=100)

# --- Morfología para reforzar silueta ---
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

# --- Inversión opcional para detector Haar/HOG (dependiendo de prueba) ---
# closed = cv2.bitwise_not(closed)

# --- Guardar imagen filtrada ---
try:
    with open(str(IMG_FILTRADA), 'wb') as f:
        _, buf = cv2.imencode('.jpg', closed)
        f.write(buf)
    print(f"Imagen preprocesada guardada correctamente en: {IMG_FILTRADA}")
except Exception as e:
    print("Error al guardar la imagen preprocesada:", e)
    exit()
