import numpy as np

def calcular_matriz_ART_inversa(num_pixeis):
    """
    Calcula a matriz inversa geométrica para a Técnica de Reconstrução 
    Algébrica (Onion-Peeling) em geometrias axissimétricas.
    
    num_pixeis: Número de pixéis do eixo de simetria até à borda (ex: 256).
    """
    A = np.zeros((num_pixeis, num_pixeis), dtype=np.float64)
    
    # Construção da matriz geométrica superior triangular
    for i in range(num_pixeis):
        for j in range(i, num_pixeis):
            # Limites dos anéis (j e j+1) em relação à linha de visão i
            r_j = j
            r_j_mais_1 = j + 1
            x_i = i
            
            # Área/Comprimento da corda no anel j intersetada pela linha x_i
            A[i, j] = 2 * (np.sqrt(r_j_mais_1**2 - x_i**2) - np.sqrt(r_j**2 - x_i**2))
            
    # Inverter a matriz uma única vez
    A_inv = np.linalg.inv(A)
    return A_inv