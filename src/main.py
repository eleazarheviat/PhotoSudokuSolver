import numpy as np
import cv2
import time
import torch
import torch.nn as nn
import os

from perspective import warp_perspective      # APLICA PERSPECTIVA AL SUDOKU DETECTADO
from split_cells import split_into_cells      # CREA LAS DIVISIONES DE TODAS LAS CELDAS
from clean_cell import preprocess_cell        # APLICA UMBRALIZACION A CADA CELDA PARA MEJOR DETECCION
from clean_cell_2 import add_white_borders    # LIMPIA LOS BORDES DE CADA UNA DE LAS CELDAS
from draw_sudoku import draw_sudoku_solution  # DIBUJA LOS RESULTADOS SOBRE IMAGEN "WARPED"





# Cargar modelo CNN entrenado

class EnhancedDigitCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),   nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),  nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),
            
            nn.Conv2d(32, 64, 3, padding=1),  nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),  nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),
            
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
           nn.Dropout(0.3),
            
            nn.Flatten(),
            nn.Linear(128 * 12 * 12, 256), nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

model = EnhancedDigitCNN()
model.load_state_dict(torch.load("modelo_digitos_50x50.pth", map_location='cpu', weights_only = True))
model.eval()                   




def process_sudoku(image_path):
    """
    Procesa una imagen de Sudoku completo: detecta el tablero, rectifica la perspectiva,
    extrae las celdas y reconoce los dígitos usando un modelo CNN.
    
    Args:
        image_path (str): Ruta de la imagen del tablero de Sudoku
        
    Returns:
        tuple: (imagen_rectificada, lista_celdas, matriz_9x9) o (None, None, None) si no detecta 4 esquinas
        
    Raises:
        Exception: Si la imagen no puede ser cargada o procesada correctamente
    """
    # 1. Cargar y redimensionar imagen
    image = cv2.imread(image_path)
    image = cv2.resize(image, (500, 800))  
    
    # 2. Preprocesamiento
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # 3. Detectar contorno del Sudoku
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    if len(approx) == 4:
        cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
        cv2.imshow("Contorno Detectado", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 5. Transformación de perspectiva
        warped = warp_perspective(image, approx)
        cv2.imshow("Sudoku Rectificado", warped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 6. Dividir en celdas
        cells = split_into_cells(warped)
        print(f"Total de celdas extraídas: {len(cells)}")
        
        sudoku = np.zeros((9,9), dtype=int)
        l = []
        
        inicio = time.time()
        
        
        for i in range(0, 81):
            cell = cells[i]
            cell_processed = preprocess_cell(cell)
            
            # ===BORRAR=== IMAGEN PARA GUARDAR ===BORRAR=== #
            para_guardar = cv2.resize(cell_processed, (50, 50))
            img_with_borders = add_white_borders(para_guardar)
            
            img = cv2.resize(img_with_borders, (50, 50))
            imagen_numpy = np.array([img])
            
            imagen_numpy = np.expand_dims(imagen_numpy, axis = 1)
            imagen_tensor = torch.from_numpy(imagen_numpy).float() / 255
    
            # Predicción con CNN
            with torch.no_grad():
                output = model(imagen_tensor)
                probabilidades = torch.nn.functional.softmax(output, dim = 1)
                prediccion = torch.argmax(output, dim = 1).item()
                confianza = probabilidades[0][prediccion].item()
                
            print(f"El modelo predice: {prediccion} con confianza de {round(confianza, 3)}")
    
            l.append(prediccion)
            
        fin = time.time()
        print('\n Tiempo de predicción total: ', fin - inicio, "s")
        
        it = 0
        for p in range(9):
            for q in range(9):
                sudoku[p, q] = int(l[it])
                it += 1
                
        #print(sudoku)
        return warped, cells, sudoku
    else:
        print("No se detectó un Sudoku válido (4 esquinas no encontradas).")
        return None, None, None
    
    
    
    
# ------------USO------------

image_path = r'C:\Users\ehevi\Desktop\imagenes_sudoku\32.jpg' # (DIRECTORIO DE LA IMAGEN A DETECTAR)
warped, cells, sudoku = process_sudoku(image_path)


from solver_sudoku import resolver_sudoku_constraint
from solver_sudoku import imprimir_tablero
from solver_sudoku import imprimir_matriz

ok, solucion = resolver_sudoku_constraint(sudoku)
if ok:
    #print("\nSudoku resuelto (formato matricial):")
    #imprimir_matriz(solucion)
    print("\nSudoku resuelto (formato con separadores):")
    imprimir_tablero(solucion)
        
else:
    print("No se pudo resolver el sudoku.")
    
cv2.imshow('SUDOKU', warped)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Cargar imagen del sudoku
sudoku_image = warped

# Opción 1: Solo números resueltos (en celdas vacías)
result1 = draw_sudoku_solution(sudoku_image, sudoku, solucion)

# Mostrar resultados
cv2.imshow('Solucion Sudoku', result1)
cv2.waitKey(0)
cv2.destroyAllWindows()
#cv2.imshow('Solucion Avanzada', result2)

# MOSTRAR RESULTADOS COMBINADOS
combinate = np.concatenate((warped, result1), axis = 1)
cv2.imshow('Solucion combinada', combinate)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Guardar resultados
#cv2.imwrite('sudoku_solucion.jpg', result1)
