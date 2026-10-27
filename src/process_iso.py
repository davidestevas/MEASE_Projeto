import numpy as np
import glob
import imageio.v3 as iio
import os
from utils import calcular_matriz_ART_inversa

# ==========================================
# 1. Configurations and Paths
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_BG = os.path.join(BASE_DIR, 'data', 'raw', 'Background', 'tiff', '*.tif')
DIR_ISO = os.path.join(BASE_DIR, 'data', 'raw', 'ISO', '*.tif') # <--- Changed to ISO
DIR_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')

os.makedirs(DIR_PROCESSED, exist_ok=True)

def calcular_background_medio(padrao_arquivos):
    arquivos = glob.glob(padrao_arquivos)
    n_frames = len(arquivos)
    print(f"Calculating mean background ({n_frames} images)...")
    soma_bg = np.zeros((512, 512), dtype=np.float64)
    for arquivo in arquivos:
        soma_bg += iio.imread(arquivo).astype(np.float64)
    return soma_bg / n_frames

def aplicar_tomografia(imagem_metade, A_inv):
    return imagem_metade @ A_inv.T

# ==========================================
# 2. Execution (ISO)
# ==========================================
bg_medio = calcular_background_medio(DIR_BG)
arquivos_iso = glob.glob(DIR_ISO)
n_iso = len(arquivos_iso)

print(f"Starting processing of {n_iso} ISO images (Simultaneous Approach 1 and 2)...")

col_simetria = 256
num_pixeis_raio = 512 - col_simetria
A_inv_precalculada = calcular_matriz_ART_inversa(num_pixeis_raio)
shape_metade = (512, num_pixeis_raio)

# Accumulators Approach 1 (Tomo -> Stats)
soma_tomo = np.zeros(shape_metade, dtype=np.float64)
soma_tomo_sq = np.zeros(shape_metade, dtype=np.float64)

# Accumulators Approach 2 (Stats -> Tomo)
soma_raw = np.zeros(shape_metade, dtype=np.float64)
soma_raw_sq = np.zeros(shape_metade, dtype=np.float64)

for i, arquivo in enumerate(arquivos_iso):
    if (i + 1) % 500 == 0:
        print(f"  -> Processed {i + 1}/{n_iso} images...")
        
    img = iio.imread(arquivo).astype(np.float64)
    img_sub = img - bg_medio
    #img_sub[img_sub < 0] = 0 
    metade_dir = img_sub[:, col_simetria:]
    
    # Approach 1
    tomo_inst = aplicar_tomografia(metade_dir, A_inv_precalculada)
    soma_tomo += tomo_inst
    soma_tomo_sq += (tomo_inst ** 2)
    
    # Approach 2
    soma_raw += metade_dir
    soma_raw_sq += (metade_dir ** 2)

# Final calculations for ISO
media_iso_ab1 = soma_tomo / n_iso
rms_iso_ab1 = np.sqrt(np.maximum((soma_tomo_sq / n_iso) - (media_iso_ab1 ** 2), 0))

media_raw = soma_raw / n_iso
rms_raw = np.sqrt(np.maximum((soma_raw_sq / n_iso) - (media_raw ** 2), 0))
media_iso_ab2 = aplicar_tomografia(media_raw, A_inv_precalculada)
rms_iso_ab2 = aplicar_tomografia(rms_raw, A_inv_precalculada)

# Save files with 'iso_' prefix
np.save(os.path.join(DIR_PROCESSED, 'media_iso_ab1.npy'), media_iso_ab1)
np.save(os.path.join(DIR_PROCESSED, 'rms_iso_ab1.npy'), rms_iso_ab1)
np.save(os.path.join(DIR_PROCESSED, 'media_iso_ab2.npy'), media_iso_ab2)
np.save(os.path.join(DIR_PROCESSED, 'rms_iso_ab2.npy'), rms_iso_ab2)

print(f"Done! ISO case files saved in: {DIR_PROCESSED}")