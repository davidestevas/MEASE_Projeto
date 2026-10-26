import numpy as np
import imageio.v3 as iio
import glob
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')
DIR_BG = os.path.join(BASE_DIR, 'data', 'raw', 'Background', 'tiff', '*.tif')

# ==========================================
# 1. Mathematical Proof: Failure of Approach 2
# ==========================================
print("--- 1. PHYSICAL ANALYSIS OF RMS ---")
rms_ab1_chama = np.load(os.path.join(DIR_PROCESSED, 'rms_ab1.npy'))
rms_ab2_chama = np.load(os.path.join(DIR_PROCESSED, 'rms_ab2.npy'))
print(f"Approach 1 (Tomography -> Stats): Min = {np.min(rms_ab1_chama):.4f} | Max = {np.max(rms_ab1_chama):.4f}")
print(f"Approach 2 (Stats -> Tomography): Min = {np.min(rms_ab2_chama):.4f} | Max = {np.max(rms_ab2_chama):.4f}")
print("-> CONCLUSION: Approach 2 generates non-physical negative values, confirming that non-linear operators (RMS) do not commute with the inverse Abel matrix.\n")

# ==========================================
# 2. Statistical Uncertainty and Sensor Noise
# ==========================================
print("--- 2. CONVERGENCE AND NOISE ---")
N_CHAMA = 3027
media_ab1_chama = np.load(os.path.join(DIR_PROCESSED, 'media_ab1.npy'))
erro_padrao = rms_ab1_chama / np.sqrt(N_CHAMA)

mascara_sinal = media_ab1_chama > 0.05
erros_relativos = (erro_padrao[mascara_sinal] / media_ab1_chama[mascara_sinal]) * 100

print(f"Absolute Standard Error (Max): {np.max(erro_padrao):.4f}")
print(f"Max Relative Error (reactive zone boundaries): {np.max(erros_relativos):.2f}%")
print(f"Median Relative Error (typical reactive zone): {np.median(erros_relativos):.2f}%\n")

print("Calculating background noise and mean matrix (please wait a few seconds)...")
arquivos_bg = glob