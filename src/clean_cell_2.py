import numpy as np
import cv2


def add_white_borders(image):
    """
    Añade bordes blancos de 7 píxeles alrededor de una imagen para mejorar el reconocimiento.
    
    Args:
        image (numpy.ndarray): Imagen binaria de entrada (50x50 píxeles)
        
    Returns:
        numpy.ndarray: Imagen con bordes blancos añadidos en los cuatro lados
    """
    
    height, width = image.shape
    result = image.copy()
    
    # Superior: 
    result[:7, :] = 255  
    
    # Inferior: 
    result[44:, :] = 255 
    
    # Izquierdo: 
    result[:, :7] = 255  
    
    # Derecho: 
    result[:, 44:] = 255  
    
    return result