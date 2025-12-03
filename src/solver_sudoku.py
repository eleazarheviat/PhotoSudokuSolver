# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 22:41:04 2025

@author: ehevi
"""

from copy import deepcopy

CELDAS = [(r, c) for r in range(9) for c in range(9)]

def unidades_y_pares():
    filas = [[(r, c) for c in range(9)] for r in range(9)]
    cols  = [[(r, c) for r in range(9)] for c in range(9)]
    cajas = []
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            cajas.append([(r, c) for r in range(br, br+3) for c in range(bc, bc+3)])
    unidades = { (r,c): [] for r,c in CELDAS }
    for u in filas + cols + cajas:
        for celda in u:
            unidades[celda].append(u)
    pares = {}
    for celda in CELDAS:
        p = set()
        for u in unidades[celda]:
            p.update(u)
        p.discard(celda)
        pares[celda] = p
    return unidades, pares

UNIDADES, PARES = unidades_y_pares()

def inicializar_dominios(tablero):
    dominios = {}
    for r, c in CELDAS:
        v = tablero[r][c]
        dominios[(r, c)] = {v} if v != 0 else set(range(1, 10))
    return dominios

def asignar(dominios, celda, valor):
    otros = dominios[celda] - {valor}
    for v in list(otros):
        if not eliminar(dominios, celda, v):
            return False
    return True

def eliminar(dominios, celda, valor):
    if valor not in dominios[celda]:
        return True
    dominios[celda].remove(valor)
    if len(dominios[celda]) == 0:
        return False
    if len(dominios[celda]) == 1:
        v = next(iter(dominios[celda]))
        for p in PARES[celda]:
            if not eliminar(dominios, p, v):
                return False
    for u in UNIDADES[celda]:
        conteo = {}
        for (r, c) in u:
            for v in dominios[(r, c)]:
                conteo[v] = conteo.get(v, 0) + 1
        for v, cnt in conteo.items():
            if cnt == 1:
                objetivo = None
                for (r, c) in u:
                    if v in dominios[(r, c)]:
                        objetivo = (r, c)
                        break
                if objetivo is not None:
                    if not asignar(dominios, objetivo, v):
                        return False
    return True

def seleccionar_celda_MRV(dominios):
    no_asignadas = [c for c in CELDAS if len(dominios[c]) > 1]
    if not no_asignadas:
        return None
    return min(no_asignadas, key=lambda c: len(dominios[c]))

def dominios_a_tablero(dominios):
    tablero = [[0]*9 for _ in range(9)]
    for (r, c) in CELDAS:
        if len(dominios[(r, c)]) != 1:
            return None
        tablero[r][c] = next(iter(dominios[(r, c)]))
    return tablero

def resolver_sudoku_constraint(tablero):
    dominios = inicializar_dominios(tablero)
    for r, c in CELDAS:
        if tablero[r][c] != 0:
            if not asignar(dominios, (r, c), tablero[r][c]):
                return False, None
    def backtrack(doms):
        celda = seleccionar_celda_MRV(doms)
        if celda is None:
            final = dominios_a_tablero(doms)
            return final is not None, final
        for v in list(doms[celda]):
            copia = deepcopy(doms)
            if asignar(copia, celda, v):
                ok, sol = backtrack(copia)
                if ok:
                    return True, sol
        return False, None
    return backtrack(dominios)

def imprimir_tablero(tablero):
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("-" * 21)
        for j in range(9):
            if j % 3 == 0 and j != 0:
                print("|", end=" ")
            print(tablero[i][j], end=" ")
        print()

def imprimir_matriz(tablero):
    for fila in tablero:
        print(fila)

# ===== Ejemplo =====
if __name__ == "__main__":
    sudoku = [[7, 4, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 8, 0, 0, 0, 0, 0],
              [2, 0, 8, 3, 4, 0, 0, 0, 7],
              [0, 0, 9, 0, 3, 0, 8, 0, 0],
              [0, 0, 0, 2, 0, 0, 5, 0, 0],
              [0, 7, 0, 0, 0, 6, 0, 0, 1],
              [1, 0, 0, 0, 8, 0, 0, 0, 9],
              [4, 9, 0, 0, 0, 0, 0, 3, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 6]]

    print("Sudoku original (formato con separadores):")
    imprimir_tablero(sudoku)
    print("\nSudoku original (formato matricial):")
    imprimir_matriz(sudoku)

    ok, solucion = resolver_sudoku_constraint(sudoku)
    if ok:
        print("\nSudoku resuelto (formato matricial):")
        imprimir_matriz(solucion)
        print("\nSudoku resuelto (formato con separadores):")
        imprimir_tablero(solucion)
        
    else:
        print("No se pudo resolver el sudoku.")
