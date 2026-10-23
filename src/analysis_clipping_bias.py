import numpy as np
import glob
import imageio.v3 as iio
import os
from utils import calcular_matriz_ART_inversa
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_BG = os.path.join(BASE_DIR, 'data', 'raw', 'Background', 'tiff', '*.tif')
DIR_CHAMA = os.path.join(BASE_DIR, 'data', 'raw', 'Chama', '*.tif')
DIR_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')

print("--- EVALUATING ZERO-CLIPPING BIAS ON MEAN FIELD ---")

# 1. Background
arquivos_bg = glob.glob(DIR_BG)
soma_bg = np.zeros((512, 512), dtype=np.float64)
for arquivo in arquivos_bg:
    soma_bg += iio.imread(arquivo).astype(np.float64)
bg_medio = soma_bg / len(arquivos_bg)

# 2. Tomografia
col_simetria = 256
num_pixeis_raio = 512 - col_simetria
A_inv = calcular_matriz_ART_inversa(num_pixeis_raio)

# 3. Processamento Sem Clipping
arquivos_chama = glob.glob(DIR_CHAMA)
n_chama = len(arquivos_chama)
soma_tomo_sem_clipping = np.zeros((512, num_pixeis_raio), dtype=np.float64)

print(f"Processing {n_chama} frames WITHOUT zero-clipping...")
for i, arquivo in enumerate(arquivos_chama):
    if (i+1) % 500 == 0: print(f"  -> {i+1}/{n_chama}")
    img = iio.imread(arquivo).astype(np.float64)
    img_sub = img - bg_medio
    # A diferença crítica: comentamos o clipping a zero
    # img_sub[img_sub < 0] = 0 
    
    metade_dir = img_sub[:, col_simetria:]
    soma_tomo_sem_clipping += metade_dir @ A_inv.T

media_sem_clipping = soma_tomo_sem_clipping / n_chama

# 4. Avaliação Comparativa
media_com_clipping = np.load(os.path.join(DIR_PROCESSED, 'media_ab1.npy'))
diferenca_absoluta = np.abs(media_com_clipping - media_sem_clipping)

mascara = media_com_clipping > 0.05
bias_relativo = (diferenca_absoluta[mascara] / media_com_clipping[mascara]) * 100

print(f"\nMax Absolute Bias from clipping: {np.max(diferenca_absoluta):.4f}")
print(f"Median Relative Bias in reactive zone: {np.median(bias_relativo):.2f}%")
print(f"Max Relative Bias in reactive zone: {np.max(bias_relativo):.2f}%")

# 5. Diagnóstico espacial do bias
np.save(os.path.join(DIR_PROCESSED, 'media_sem_clipping.npy'), media_sem_clipping)

idx = np.unravel_index(np.argmax(diferenca_absoluta), diferenca_absoluta.shape)
print(f"Local do bias máximo: Z={idx[0]}, R={idx[1]}")
print(f"  com clipping: {media_com_clipping[idx]:.3f} | sem clipping: {media_sem_clipping[idx]:.3f}")

# Bias apenas no núcleo de alto sinal (20% do máximo) - métrica honesta
mascara_nucleo = media_com_clipping > 0.2 * media_com_clipping.max()
bias_nucleo = (diferenca_absoluta[mascara_nucleo] / media_com_clipping[mascara_nucleo]) * 100
print(f"Bias relativo MEDIANO no núcleo de alto sinal: {np.median(bias_nucleo):.1f}%")

# Figura lado a lado
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
for ax, campo, titulo in zip(axs, [media_com_clipping, media_sem_clipping],
                              ['Mean Field (with zero-clipping)', 'Mean Field (WITHOUT clipping)']):
    im = ax.imshow(campo, cmap='RdBu_r', vmin=-np.max(np.abs(media_sem_clipping)),
                   vmax=np.max(np.abs(media_sem_clipping)))
    ax.set_title(titulo); fig.colorbar(im, ax=ax, fraction=0.046)
plt.savefig(os.path.join(DIR_PROCESSED, 'clipping_comparison.png'), dpi=300, bbox_inches='tight')