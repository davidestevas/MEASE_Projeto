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

arquivos_bg = glob.glob(DIR_BG)
soma_bg = np.zeros((512, 512), dtype=np.float64)
for arquivo in arquivos_bg:
    soma_bg += iio.imread(arquivo).astype(np.float64)
bg_medio = soma_bg / len(arquivos_bg)

col_simetria = 256
num_pixeis_raio = 512 - col_simetria
A_inv = calcular_matriz_ART_inversa(num_pixeis_raio)

arquivos_chama = glob.glob(DIR_CHAMA)
n_chama = len(arquivos_chama)

soma_tomo_sem_clipping = np.zeros((512, num_pixeis_raio), dtype=np.float64)
soma_tomo_com_clipping = np.zeros((512, num_pixeis_raio), dtype=np.float64)

print(f"Processing {n_chama} frames to compare clipping impact...")
for i, arquivo in enumerate(arquivos_chama):
    if (i+1) % 500 == 0: print(f"  -> {i+1}/{n_chama}")
    img = iio.imread(arquivo).astype(np.float64)
    
    # 1. Sem Clipping (Físico Real)
    img_sub_sem = img - bg_medio
    soma_tomo_sem_clipping += (img_sub_sem[:, col_simetria:] @ A_inv.T)
    
    # 2. Com Clipping (Simulação do Erro)
    img_sub_com = np.copy(img_sub_sem)
    img_sub_com[img_sub_com < 0] = 0
    soma_tomo_com_clipping += (img_sub_com[:, col_simetria:] @ A_inv.T)

media_sem_clipping = soma_tomo_sem_clipping / n_chama
media_com_clipping = soma_tomo_com_clipping / n_chama

diferenca_absoluta = np.abs(media_com_clipping - media_sem_clipping)
mascara = media_com_clipping > 0.05
bias_relativo = (diferenca_absoluta[mascara] / media_com_clipping[mascara]) * 100

print(f"\nMax Absolute Bias from clipping: {np.max(diferenca_absoluta):.4f}")
print(f"Median Relative Bias in reactive zone: {np.median(bias_relativo):.2f}%")
print(f"Max Relative Bias in reactive zone: {np.max(bias_relativo):.2f}%")

idx = np.unravel_index(np.argmax(diferenca_absoluta), diferenca_absoluta.shape)
print(f"Location of maximum bias: Z={idx[0]}, R={idx[1]}")
print(f"  with clipping: {media_com_clipping[idx]:.3f} | without clipping: {media_sem_clipping[idx]:.3f}")

mascara_nucleo = media_com_clipping > 0.2 * media_com_clipping.max()
bias_nucleo = (diferenca_absoluta[mascara_nucleo] / media_com_clipping[mascara_nucleo]) * 100
print(f"MEDIAN relative bias in the high-signal core: {np.median(bias_nucleo):.1f}%")

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
for ax, campo, titulo in zip(axs, [media_com_clipping, media_sem_clipping],
                              ['Mean Field (with zero-clipping)', 'Mean Field (WITHOUT clipping)']):
    im = ax.imshow(campo, cmap='RdBu_r', vmin=-np.max(np.abs(media_sem_clipping)),
                   vmax=np.max(np.abs(media_sem_clipping)))
    ax.set_title(titulo); fig.colorbar(im, ax=ax, fraction=0.046)
plt.savefig(os.path.join(DIR_PROCESSED, 'clipping_comparison.png'), dpi=300, bbox_inches='tight')