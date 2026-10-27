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
    print(f"Calculating mean background ({n_frames} images)...")
    soma_bg = np.zeros((512, 512), dtype=np.float64)
    for arquivo in arquivos:
        soma_bg += iio.imread(arquivo).astype(np.float64)
    return soma_bg / n_frames

def aplicar_tomografia(imagem_metade, A_inv):
    return imagem_metade @ A_inv.T

def extrair_metade_direita(img_sub, col_simetria, num_pixeis_raio, delta):
    """Extracts the right half of the image, applying sub-pixel shift if |delta| > 0.25 px."""
    if abs(delta) <= 0.25:
        return img_sub[:, col_simetria:]
    c_float = col_simetria + delta + np.arange(num_pixeis_raio, dtype=np.float64)
    c0 = np.clip(np.floor(c_float).astype(int), 0, 511)
    c1 = np.clip(c0 + 1, 0, 511)
    w = c_float - np.floor(c_float)
    return (1.0 - w) * img_sub[:, c0] + w * img_sub[:, c1]

# ==========================================
# 2. Pass 1: Full Raw Mean & Symmetry Axis Measurement (Point 3)
# ==========================================
bg_medio = calcular_background_medio(DIR_BG)
arquivos_chama = glob.glob(DIR_CHAMA)
n_chama = len(arquivos_chama)

col_simetria = 256
num_pixeis_raio = 512 - col_simetria
A_inv_precalculada = calcular_matriz_ART_inversa(num_pixeis_raio)

print(f"Pass 1/2: Accumulating full 512x512 raw mean over {n_chama} flame frames...")
soma_raw_full = np.zeros((512, 512), dtype=np.float64)
for i, arquivo in enumerate(arquivos_chama):
    if (i + 1) % 500 == 0:
        print(f"  -> Pass 1: {i + 1}/{n_chama} images...")
    img = iio.imread(arquivo).astype(np.float64)
    soma_raw_full += (img - bg_medio)

media_raw_full = soma_raw_full / n_chama
np.save(os.path.join(DIR_PROCESSED, 'media_raw_full.npy'), media_raw_full)

# Measure symmetry axis offset delta by minimizing |M(x0 + r) - M(x0 - r)| in the flow domain
Z_FLOW = media_raw_full[20:300, :]
r_test = np.arange(5, 180, dtype=np.float64)
x_grid = np.arange(512, dtype=np.float64)
deltas_grid = np.linspace(-5.0, 5.0, 1001)  # 0.01 px resolution
erros_assimetria = np.zeros_like(deltas_grid)

for idx_d, d in enumerate(deltas_grid):
    x0 = col_simetria + d
    dir_interp = np.array([np.interp(x0 + r_test, x_grid, row) for row in Z_FLOW])
    esq_interp = np.array([np.interp(x0 - r_test, x_grid, row) for row in Z_FLOW])
    erros_assimetria[idx_d] = np.mean(np.abs(dir_interp - esq_interp))

delta_medido = float(deltas_grid[np.argmin(erros_assimetria)])
delta_aplicado = delta_medido if abs(delta_medido) > 0.25 else 0.0
np.save(os.path.join(DIR_PROCESSED, 'delta_eixo.npy'), np.array([delta_medido, delta_aplicado]))

print(f"Measured Symmetry Axis Offset: delta = {delta_medido:+.3f} px (Center col = {col_simetria + delta_medido:.3f})")
if abs(delta_medido) > 0.25:
    print(f"  -> |delta| > 0.25 px: Applying sub-pixel shift ({delta_aplicado:+.3f} px) during reconstruction!")
else:
    print("  -> |delta| <= 0.25 px: Nominal axis (col 256) confirmed within tolerance.")

# ==========================================
# 3. Pass 2: Simultaneous Approach 1 and Approach 2
# ==========================================
shape_metade = (512, num_pixeis_raio)
soma_tomo = np.zeros(shape_metade, dtype=np.float64)
soma_tomo_sq = np.zeros(shape_metade, dtype=np.float64)
soma_raw = np.zeros(shape_metade, dtype=np.float64)
soma_raw_sq = np.zeros(shape_metade, dtype=np.float64)

print(f"Pass 2/2: Processing {n_chama} flame frames (Approach 1 & Approach 2)...")
for i, arquivo in enumerate(arquivos_chama):
    if (i + 1) % 500 == 0:
        print(f"  -> Pass 2: {i + 1}/{n_chama} images...")
    img = iio.imread(arquivo).astype(np.float64)
    img_sub = img - bg_medio
    metade_dir = extrair_metade_direita(img_sub, col_simetria, num_pixeis_raio, delta_aplicado)
    
    # Approach 1 (Tomo -> Stats)
    tomo_inst = aplicar_tomografia(metade_dir, A_inv_precalculada)
    soma_tomo += tomo_inst
    soma_tomo_sq += (tomo_inst ** 2)
    
    # Approach 2 (Stats -> Tomo)
    soma_raw += metade_dir
    soma_raw_sq += (metade_dir ** 2)

media_ab1 = soma_tomo / n_chama
rms_ab1 = np.sqrt(np.maximum((soma_tomo_sq / n_chama) - (media_ab1 ** 2), 0))

media_raw = soma_raw / n_chama
rms_raw = np.sqrt(np.maximum((soma_raw_sq / n_chama) - (media_raw ** 2), 0))
media_ab2 = aplicar_tomografia(media_raw, A_inv_precalculada)
rms_ab2 = aplicar_tomografia(rms_raw, A_inv_precalculada)

np.save(os.path.join(DIR_PROCESSED, 'media_ab1.npy'), media_ab1)
np.save(os.path.join(DIR_PROCESSED, 'rms_ab1.npy'), rms_ab1)
np.save(os.path.join(DIR_PROCESSED, 'media_ab2.npy'), media_ab2)
np.save(os.path.join(DIR_PROCESSED, 'rms_ab2.npy'), rms_ab2)

print(f"Done! Flame files saved in: {DIR_PROCESSED}")