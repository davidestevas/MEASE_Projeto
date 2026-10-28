import os
import numpy as np

def calcular_matriz_ART_inversa(num_pixeis):
    """
    Calculates the inverse geometric matrix for the Algebraic Reconstruction Technique
    (Onion-Peeling) in axisymmetric geometries.
    Note: The index j in {0, ..., N-1} represents the ring [j, j+1] with physical center at r = j + 0.5 px.
    """
    A = np.zeros((num_pixeis, num_pixeis), dtype=np.float64)
    for i in range(num_pixeis):
        for j in range(i, num_pixeis):
            r_j = float(j)
            r_j_mais_1 = float(j + 1)
            x_i = float(i)
            A[i, j] = 2.0 * (np.sqrt(r_j_mais_1**2 - x_i**2) - np.sqrt(r_j**2 - x_i**2))
    return np.linalg.inv(A)

def extrair_metade_direita(img_sub, col_simetria=256, num_pixeis_raio=256, delta=0.0):
    """
    Extracts the right half of the image from the real axis (col_simetria + delta),
    using sub-pixel linear interpolation when |delta| > 0.25 px.
    """
    if abs(delta) <= 0.25:
        return img_sub[:, col_simetria:col_simetria + num_pixeis_raio]
    c_float = (col_simetria + delta) + np.arange(num_pixeis_raio, dtype=np.float64)
    c0 = np.clip(np.floor(c_float).astype(int), 0, img_sub.shape[1] - 1)
    c1 = np.clip(c0 + 1, 0, img_sub.shape[1] - 1)
    w = c_float - np.floor(c_float)
    return (1.0 - w) * img_sub[:, c0] + w * img_sub[:, c1]

def carregar_delta_eixo(dir_processed):
    """Loads (delta_measured, delta_applied) from delta_eixo.npy."""
    caminho = os.path.join(dir_processed, 'delta_eixo.npy')
    if os.path.exists(caminho):
        dados = np.load(caminho)
        return float(dados[0]), float(dados[1])
    return 0.0, 0.0

def medir_corpo_rombo_background(bg_medio, col_nominal=256):
    """
    Measures the diameter in pixels (D_b_px) and the geometric center of the burner
    in the lower region of the background image (Z = 335..500 px) with
    sub-pixel parabolic refinement: idx + 0.5*(y0 - y2)/(y0 - 2*y1 + y2).
    """
    perfil_x = np.mean(bg_medio[335:500, :], axis=0)
    grad_x = np.abs(np.gradient(perfil_x))
    
    x_esq = int(np.argmax(grad_x[180:240]) + 180)
    x_dir = int(np.argmax(grad_x[280:340]) + 280)
    
    def refinar_pico(arr, idx):
        if 1 <= idx < len(arr) - 1:
            y0, y1, y2 = float(arr[idx - 1]), float(arr[idx]), float(arr[idx + 1])
            denom = y0 - 2.0 * y1 + y2
            if abs(denom) > 1e-12:
                return idx + 0.5 * (y0 - y2) / denom
        return float(idx)
    
    x_esq_sub = refinar_pico(grad_x, x_esq)
    x_dir_sub = refinar_pico(grad_x, x_dir)
    
    d_bluff_px = x_dir_sub - x_esq_sub
    centro_bg = 0.5 * (x_esq_sub + x_dir_sub)
    delta_bg = centro_bg - col_nominal
    return d_bluff_px, centro_bg, delta_bg