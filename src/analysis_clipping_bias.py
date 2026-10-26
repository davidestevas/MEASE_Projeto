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

# Load unclipped mean field directly from media_ab1.npy (already computed without clipping)
caminho_ab1 = os.path.join(DIR_PROCESSED, 'media_ab1.npy')
if os.path.exists(caminho_ab1):
    print("Loading unclipped mean field from media_ab1.npy...")
    media_sem_clipping = np.load(caminho_ab1)
    calcular_sem = False
else:
    soma_tomo_sem_clipping = np.zeros((512, num_pixeis_raio), dtype=np.float64)
    calcular_sem = True

soma_tomo_com_clipping = np.zeros((512, num_pixeis_raio), dtype=np.float64)

print(f"Processing {n_chama} frames to simulate zero-clipping impact...")
for i, arquivo in enumerate(arquivos_chama):
    if (i + 1) % 500 == 0:
        print(f"  -> {i + 1}/{n_chama}")
    img = iio.imread(arquivo).astype(np.float64)
    img_sub_sem = img - bg_medio
    
    if calcular_sem:
        soma_tomo_sem_clipping += (img_sub_sem[:, col_simetria:] @ A_inv.T)
    
    # Com Clipping (Simulação do Erro)
    img_sub_com = np.maximum(img_sub_sem, 0)
    soma_tomo_com_clipping += (img_sub_com[:, col_simetria:] @ A_inv.T)

if calcular_sem:
    media_sem_clipping = soma_tomo_sem_clipping / n_chama
media_com_clipping = soma_tomo_com_clipping / n_chama

# 4. Comparative Evaluation (Restricted to flow domain Z < 325 px, excluding bluff-body)
Z_MASK = 325
media_sem_flow = media_sem_clipping[:Z_MASK, :].copy()
media_com_flow = media_com_clipping[:Z_MASK, :].copy()

diferenca_absoluta = np.abs(media_com_flow - media_sem_flow)
norma_sem = np.linalg.norm(media_sem_flow)
norma_com = np.linalg.norm(media_com_flow)

print(f"\nL2 Norm of Mean Field (Z < {Z_MASK}): Without Clipping = {norma_sem:.4f} | With Clipping = {norma_com:.4f} ({(norma_com/norma_sem)*100:.1f}%)")
print(f"Max Absolute Bias in flow domain: {np.max(diferenca_absoluta):.4f}")

idx = np.unravel_index(np.argmax(diferenca_absoluta), diferenca_absoluta.shape)
print(f"Location of maximum flow bias: Z={idx[0]}, R={idx[1]}")
print(f"  with clipping: {media_com_flow[idx]:.3f} | without clipping: {media_sem_flow[idx]:.3f}")

# Relative bias in the high-signal region (|signal| > 20% of 99th percentile of unclipped field)
lim_sinal = 0.20 * np.percentile(np.abs(media_sem_flow), 99)
mascara_nucleo = np.abs(media_sem_flow) > lim_sinal
bias_nucleo = (diferenca_absoluta[mascara_nucleo] / np.abs(media_sem_flow[mascara_nucleo])) * 100
print(f"MEDIAN relative bias in active Schlieren zone: {np.median(bias_nucleo):.1f}%")

# Side-by-side figure with bluff-body masked
plot_com = media_com_clipping.copy()
plot_sem = media_sem_clipping.copy()
plot_com[Z_MASK:, :] = 0
plot_sem[Z_MASK:, :] = 0
lim_cor = np.percentile(np.abs(media_sem_flow), 99)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
for ax, campo, titulo in zip(axs, [plot_com, plot_sem],
                              ['Mean Schlieren (WITH zero-clipping)', 'Mean Schlieren (WITHOUT clipping)']):
    im = ax.imshow(campo, cmap='RdBu_r', vmin=-lim_cor, vmax=lim_cor, aspect='auto')
    ax.axhline(Z_MASK, color='black', linestyle='--', alpha=0.5)
    ax.set_title(titulo)
    ax.set_xlabel('Radius (px)')
    ax.set_ylabel('Z Height (px)')
    fig.colorbar(im, ax=ax, fraction=0.046)

caminho_out = os.path.join(DIR_PROCESSED, 'clipping_comparison.png')
plt.savefig(caminho_out, dpi=300, bbox_inches='tight')
print(f"Saved figure: {caminho_out}")