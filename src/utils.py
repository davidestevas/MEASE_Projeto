import numpy as np

def calcular_matriz_ART_inversa(num_pixeis):
    """
    Calculates the geometric inverse matrix for the Algebraic Reconstruction 
    Technique (Onion-Peeling) in axisymmetric geometries.
    
    num_pixeis: Number of pixels from the symmetry axis to the edge (e.g., 256).
    """
    A = np.zeros((num_pixeis, num_pixeis), dtype=np.float64)
    
    # Construction of the upper triangular geometric matrix
    for i in range(num_pixeis):
        for j in range(i, num_pixeis):
            # Boundaries of the rings (j and j+1) relative to the line of sight i
            r_j = j
            r_j_mais_1 = j + 1
            x_i = i
            
            # Area/Length of the chord in ring j intersected by line x_i
            A[i, j] = 2 * (np.sqrt(r_j_mais_1**2 - x_i**2) - np.sqrt(r_j**2 - x_i**2))
            
    # Invert the matrix a single time
    A_inv = np.linalg.inv(A)
    return A_inv