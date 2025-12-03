import numpy as np
import cv2


def draw_sudoku_solution(original_image, sudoku_grid, solved_grid):
    """
    Dibuja los números del sudoku resuelto centrados en cada celda vacía original.
    
    Args:
        original_image (numpy.ndarray): Imagen original del sudoku (450x450)
        sudoku_grid (numpy.ndarray): Matriz 9x9 con los números originales (0 para vacíos)
        solved_grid (numpy.ndarray): Matriz 9x9 con la solución completa
        
    Returns:
        numpy.ndarray: Imagen anotada con los números de la solución en color naranja,
                     superpuestos sobre la imagen original del tablero
    """
    # Crear una copia de la imagen original
    annotated_image = original_image.copy()
    
    # Tamaño de cada celda (450/9 = 50)
    cell_size = 50
    
    # Configuración del texto
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 3
    color_solution = (0, 165, 255)  # Naranja para números resueltos
    color_original = (0, 255, 0)    # Verde para números originales
    
    for row in range(9):
        for col in range(9):
            # Calcular posición del centro de la celda
            center_x = col * cell_size + cell_size // 2
            center_y = row * cell_size + cell_size // 2
            
            # Obtener número original y resuelto
            original_num = sudoku_grid[row][col]
            solved_num = solved_grid[row][col]
            
            # Solo dibujar números en celdas vacías originalmente
            if original_num == 0 and solved_num != 0:
                # Convertir número a string
                num_str = str(solved_num)
                
                # Calcular tamaño del texto para centrarlo
                text_size = cv2.getTextSize(num_str, font, font_scale, thickness)[0]
                text_x = center_x - text_size[0] // 2
                text_y = center_y + text_size[1] // 2
                
                # Dibujar texto centrado
                cv2.putText(annotated_image, num_str, (text_x, text_y), 
                           font, font_scale, color_solution, thickness)
    
    return annotated_image
