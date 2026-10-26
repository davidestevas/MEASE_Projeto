import numpy as np
import glob
import imageio.v3 as iio
import os
from utils import calcular_matriz_ART_inversa

# ==========================================
# 1. Configurations and Absolute Paths
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_BG = os.path.join(BASE_DIR, 'data', 'raw', 'Background', 'tiff', '*.tif')
DIR_CHAMA = os.path.join(BASE_DIR, 'data', 'raw', 'Chama', '*.tif')
DIR_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')

os.makedirs(DIR_PROCESSED, exist_ok=True)

def calcular_background_medio(padrao_arquivos):
    arquivos = glob.glob(padrao_arquivos)
    n_frames = len(arquivos)
    if n_frames == 0:
        raise ValueError(f"ERROR: No image found in the path {padrao_arquivos}.")
    soma_bg = np.zeros((512, 512), dtype=np.float64)
    for arquivo in arquivos:
        soma_bg += iio.imread(arquivo).astype(np.float64)
    return soma_bg / n_frames

def aplicar_tomografia(imagem_metade, A_inv):
    return imagem_metade @ A_inv.T

# ==========================================
# 2. Execution of Approach 1 (Tomo -> Mean/RMS)
# ==========================================
bg_medio = calcular_background_medio(DIR_BG)
arquivos_chama = glob.glob(DIR_CHAMA)
n_chama = len(arquivos_chama)

col_simetria = 256
num_pixeis_raio = 512 - col_simetria
A_inv_precalculada = calcular_matriz_ART_inversa(num_pixeis_raio)

shape_metade = (512, 512 - col_simetria)
soma_tomo = np.zeros(shape_metade, dtype=np.float64)
soma_tomo_sq = np.zeros(shape_metade, dtype=np.float64)

for arquivo in arquivos_chama:
    img = iio.imread(arquivo).astype(np.float64)
    img_sub = img - bg_medio
    
    # CLIPPING REMOVIDO: img_sub[img_sub < 0] = 0
    metade_dir = img_sub[:, col_simetria:]
    
    tomo_inst = aplicar_tomografia(metade_dir, A_inv_precalculada)
    soma_tomo += tomo_inst
    soma_tomo_sq += (tomo_inst ** 2)

media_ab1 = soma_tomo / n_chama
rms_ab1 = np.sqrt((soma_tomo_sq / n_chama) - (media_ab1 ** 2))

# ==========================================
# 3. Execution of Approach 2 (Mean/RMS -> Tomo)
# ==========================================
soma_raw = np.zeros(shape_metade, dtype=np.float64)
soma_raw_sq = np.zeros(shape_metade, dtype=np.float64)

for arquivo in arquivos_chama:
    img = iio.imread(arquivo).astype(np.float64)
    img_sub = img - bg_medio
    
    # CLIPPING REMOVIDO
    metade_dir = img_sub[:, col_simetria:]
    
    soma_raw += metade_dir
    soma_raw_sq += (metade_dir ** 2)

media_raw = soma_raw / n_chama
rms_raw = np.sqrt((soma_raw_sq / n_chama) - (media_raw ** 2))

media_ab2 = aplicar_tomografia(media_raw, A_inv_precalculada)
rms_ab2 = aplicar_tomografia(rms_raw, A_inv_precalculada)

np.save(os.path.join(DIR_PROCESSED, 'media_ab1.npy'), media_ab1)
np.save(os.path.join(DIR_PROCESSED, 'rms_ab1.npy'), rms_ab1)
np.save(os.path.join(DIR_PROCESSED, 'media_ab2.npy'), media_ab2)
np.save(os.path.join(DIR_PROCESSED, 'rms_ab2.npy'), rms_ab2)

print(f"Done! Files saved in the folder: {DIR_PROCESSED}")