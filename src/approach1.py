import numpy as np
import glob
import imageio.v3 as iio
import os
from utils import calcular_matriz_ART_inversa

# ==========================================
# 1. Configurations and Absolute Paths
# ==========================================
# Automatically detects the folder where this script is (src/) and goes up one level to the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DIR_BG = os.path.join(BASE_DIR, 'data', 'raw', 'Background', 'tiff', '*.tif')
DIR_CHAMA = os.path.join(BASE_DIR, 'data', 'raw', 'Chama', '*.tif')
DIR_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')

# Ensures that the destination folder exists
os.makedirs(DIR_PROCESSED, exist_ok=True)

def calcular_background_medio(padrao_arquivos):
    arquivos = glob.glob(padrao_arquivos)
    n_frames = len(arquivos)
    
    if n_frames == 0:
        raise ValueError(f"ERROR: No image found in the path {padrao_arquivos}. Confirm if you extracted the files correctly.")
        
    print(f"Calculating mean background ({n_frames} images)...")
    soma_bg = np.zeros((512, 512), dtype=np.float64)
    
    for arquivo in arquivos:
        img = iio.imread(arquivo).astype(np.float64)
        soma_bg += img
        
    print("Mean background successfully calculated!")
    return soma_bg / n_frames

def aplicar_tomografia(imagem_metade, A_inv):
    # The matrix product applies the reconstruction to all horizontal rows simultaneously
    return imagem_metade @ A_inv.T

# ==========================================
# 2. Execution of Approach 1 (Tomo -> Mean/RMS)
# ==========================================
bg_medio = calcular_background_medio(DIR_BG)
arquivos_chama = glob.glob(DIR_CHAMA)
n_chama = len(arquivos_chama)

# The symmetry axis needs to be identified (example: column 256)
col_simetria = 256
num_pixeis_raio = 512 - col_simetria
A_inv_precalculada = calcular_matriz_ART_inversa(num_pixeis_raio)

# Accumulators for Mean and RMS (only for the reconstructed half)
shape_metade = (512, 512 - col_simetria)
soma_tomo = np.zeros(shape_metade, dtype=np.float64)
soma_tomo_sq = np.zeros(shape_metade, dtype=np.float64)

for arquivo in arquivos_chama:
    # 1. Read image and convert to float
    img = iio.imread(arquivo).astype(np.float64)
    
    # 2. Subtract background (avoid spurious negative noise values)
    img_sub = img - bg_medio
    img_sub[img_sub < 0] = 0 
    
    # 3. Cut right half from the symmetry axis
    metade_dir = img_sub[:, col_simetria:]
    
    # 4. APPROACH 1: IMMEDIATE Tomography at each instant
    tomo_inst = aplicar_tomografia(metade_dir, A_inv_precalculada)
    
    # 5. Accumulate for the statistics
    soma_tomo += tomo_inst
    soma_tomo_sq += (tomo_inst ** 2)

# Final Calculations - Approach 1
media_ab1 = soma_tomo / n_chama
rms_ab1 = np.sqrt((soma_tomo_sq / n_chama) - (media_ab1 ** 2))

# ==========================================
# 3. Execution of Approach 2 (Mean/RMS -> Tomo)
# ==========================================
# Initialize accumulators directly for the subtracted raw data
soma_raw = np.zeros(shape_metade, dtype=np.float64)
soma_raw_sq = np.zeros(shape_metade, dtype=np.float64)

for arquivo in arquivos_chama:
    img = iio.imread(arquivo).astype(np.float64)
    img_sub = img - bg_medio
    img_sub[img_sub < 0] = 0
    metade_dir = img_sub[:, col_simetria:]
    
    # Accumulate STATISTICS BEFORE tomography
    soma_raw += metade_dir
    soma_raw_sq += (metade_dir ** 2)

# 1. Calculate Mean and RMS of the integrated projections (Raw)
media_raw = soma_raw / n_chama
rms_raw = np.sqrt((soma_raw_sq / n_chama) - (media_raw ** 2))

# 2. APPROACH 2: Apply SINGLE Tomography at the end
media_ab2 = aplicar_tomografia(media_raw, A_inv_precalculada)
rms_ab2 = aplicar_tomografia(rms_raw, A_inv_precalculada)

# Save results in the processed folder
caminho_media_ab1 = os.path.join(DIR_PROCESSED, 'media_ab1.npy')
caminho_rms_ab1 = os.path.join(DIR_PROCESSED, 'rms_ab1.npy')
# Paths for Approach 2
caminho_media_ab2 = os.path.join(DIR_PROCESSED, 'media_ab2.npy')
caminho_rms_ab2 = os.path.join(DIR_PROCESSED, 'rms_ab2.npy')

np.save(caminho_media_ab1, media_ab1)
np.save(caminho_rms_ab1, rms_ab1)
np.save(caminho_media_ab2, media_ab2)
np.save(caminho_rms_ab2, rms_ab2)

print(f"Done! Files saved in the folder: {DIR_PROCESSED}")