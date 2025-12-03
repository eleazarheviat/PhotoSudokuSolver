import numpy as np
import cv2

from perspective import warp_perspective  # funcion importada

def split_into_cells(warped_sudoku):
    """
    Divide una imagen rectificada de Sudoku en 81 celdas individuales.
    
    Args:
        warped_sudoku (numpy.ndarray): Imagen del tablero de Sudoku rectificado (450x450)
        
    Returns:
        list: Lista de 81 imágenes correspondientes a cada celda del tablero
    """
    cells = []
    h, w = warped_sudoku.shape[:2]
    cell_size = h // 9  
    
    for row in range(9):
        for col in range(9):
            start_x = col * cell_size
            start_y = row * cell_size
            end_x = start_x + cell_size
            end_y = start_y + cell_size
            cell = warped_sudoku[start_y:end_y, start_x:end_x]
            cells.append(cell)
    return cells