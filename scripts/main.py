import cv2
import time
from pathlib import Path
from subprocess import run

# --- Carpetas absolutas ---
ORIGINAL_DIR = Path(r"C:\Users\jgome\Desktop\ArchivosUni\Procesamiento de Imágenes\Detecci-n-Humana-en-cv2\data\original")
FILTRADA_DIR = Path(r"C:\Users\jgome\Desktop\ArchivosUni\Procesamiento de Imágenes\Detecci-n-Humana-en-cv2\data\filtrada")
DETECTADA_DIR = Path(r"C:\Users\jgome\Desktop\ArchivosUni\Procesamiento de Imágenes\Detecci-n-Humana-en-cv2\data\detectada")

# Crear carpetas si no existen
for d in [ORIGINAL_DIR, FILTRADA_DIR, DETECTADA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- Nombres de archivos por fase ---
IMG_ORIGINAL = ORIGINAL_DIR / "frame_original.jpg"
IMG_FILTRADA = FILTRADA_DIR / "frame_preprocesado.jpg"
IMG_DETECTADA = DETECTADA_DIR / "frame_detectado.jpg"

# --- Captura de cámara ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

print("Cámara activada. Esperando 3 segundos antes de tomar la foto...")
time.sleep(3)

ret, frame = cap.read()
if not ret or frame is None:
    print("No se pudo capturar un frame")
    cap.release()
    exit()

# --- Mostrar el frame para verificación ---
cv2.imshow("Frame Capturado", frame)
print("Frame capturado. Presiona cualquier tecla para continuar...")
cv2.waitKey(0)
cv2.destroyAllWindows()

# --- Guardar imagen original con patrón robusto ---
try:
    with open(str(IMG_ORIGINAL), 'wb') as f:
        _, buf = cv2.imencode('.jpg', frame)
        f.write(buf)
    print(f"Imagen guardada correctamente en: {IMG_ORIGINAL}")
except Exception as e:
    print("Error al guardar la imagen:", e)
    cap.release()
    exit()

# --- Apagar cámara ---
time.sleep(2)  # Esperar un momento antes de liberar
cap.release()
print("Cámara apagada")

# --- Ejecutar filtros.py ---
print("Ejecutando filtros.py...")
run(["python", "scripts/filtros.py"], check=True)

# --- Ejecutar detector.py ---
print("Ejecutando detector.py...")
run(["python", "scripts/detector.py"], check=True)

# --- Verificar resultado final ---
if IMG_DETECTADA.exists():
    print("Pipeline finalizado. Imagen detectada disponible en:", IMG_DETECTADA)
else:
    print("No se generó la imagen detectada.")
