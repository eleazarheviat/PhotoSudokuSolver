import numpy as np
import cv2

def warp_perspective(image, contour):
    """
    Aplica transformación de perspectiva para rectificar un tablero de Sudoku.
    
    Toma un contorno de 4 puntos y realiza una corrección de perspectiva para
    obtener una vista frontal del tablero con dimensiones fijas (450x450).
    
    Args:
        image (numpy.ndarray): Imagen original del Sudoku
        contour (numpy.ndarray): Contorno de 4 puntos que define el tablero
        
    Returns:
        numpy.ndarray: Imagen rectificada del Sudoku (450x450 píxeles)
    """
    points = contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]  
    rect[2] = points[np.argmax(s)]  
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]  
    rect[3] = points[np.argmax(diff)]  
    
    width = height = 450  
    dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (width, height))
    
    return warped
