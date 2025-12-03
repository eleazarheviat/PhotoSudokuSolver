import numpy as np
import cv2



def preprocess_cell(cell):
    """
    Preprocesa una imagen de celda de Sudoku para reconocimiento de dígitos.
    
    Convierte a escala de grises, aplica umbralización binaria con método Otsu
    y realiza operaciones morfológicas para limpiar ruido y mejorar la calidad.
    
    Args:
        cell (numpy.ndarray): Imagen de la celda en color BGR
        
    Returns:
        numpy.ndarray: Imagen binaria preprocesada lista para clasificación
    """
    
    
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return cleaned

