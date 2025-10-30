import cv2
import numpy as np

def procesar_baja_luz(frame):
    """Procesa un frame para mejorar detección en baja iluminación con reducción de ruido conservando bordes"""
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # CLAHE para mejorar contraste en baja luz
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray_clahe = clahe.apply(gray)

    # Corrección gamma (aclara zonas oscuras)
    gamma = 1.6
    lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
    bright = cv2.LUT(gray_clahe, lut)

    # Filtro bilateral → reduce ruido preservando bordes
    smooth = cv2.bilateralFilter(bright, d=7, sigmaColor=50, sigmaSpace=50)

    # Opcional: eliminación de ruido fino extra
    denoised = cv2.fastNlMeansDenoising(smooth, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # Volver a 3 canales
    final = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

    return final

# import cv2 
# import numpy as np 

# def procesar_baja_luz(frame):
#     """Procesa un frame para mejorar detección en baja iluminación"""
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Equalización de histograma
#     gray_eq = cv2.equalizeHist(gray)

#     # Corrección gamma
#     gamma = 1.8
#     lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
#     bright = cv2.LUT(gray_eq, lut)

#     # Reducir ruido
#     blur = cv2.GaussianBlur(bright, (3,3), 0)

#     # Volver a 3 canales
#     final = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)
#     return final