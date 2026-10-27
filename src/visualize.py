import numpy as np
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')

media_ab1 = np.load(os.path.join(DIR_PROCESSED, 'media_ab1.npy')).copy()
caminho_rms_corr = os.path.join(DIR_PROCESSED, 'rms_ab1_corrigido.npy')
rms_ab1 = np.load(caminho_rms_corr if os.path.exists(caminho_rms_corr) else os.path.join(DIR_PROCESSED, 'rms_ab1.npy')).copy()
media_ab2 = np.load(os.path.join(DIR_PROCESSED, 'media_ab2.npy')).copy()
rms_ab2 = np.load(os.path.join(DIR_PROCESSED, 'rms_ab2.npy')).copy()

Z_MASK = 325
for mat in (media_ab1, rms_ab1, media_ab2, rms_ab2):
    mat[Z_MASK:, :] = 0

lim_media = np.percentile(np.abs(media_ab1[:Z_MASK, :]), 99)
lim_rms = np.percentile(rms_ab1[:Z_MASK, :], 99)

fig, axs = plt.subplots(2, 2, figsize=(13, 10))

def plot_campo(ax, matriz, titulo, vmin, vmax, cmap='RdBu_r'):
    im = ax.imshow(matriz, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax.axhline(Z_MASK, color='black', linestyle='--', alpha=0.5, label='Bluff-body Mask')
    ax.set_title(titulo, fontsize=11.5)
    ax.set_xlabel('Radius r (px) [R_bluff ≈ 58 px]')
    ax.set_ylabel('Axial Coordinate z (px)')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'Reconstructed $\partial n / \partial z$ (a.u.)')

plot_campo(axs[0, 0], media_ab1, "Flame Mean (Approach 1: Tomo -> Stats)", -lim_media, lim_media, cmap='RdBu_r')
plot_campo(axs[0, 1], media_ab2, "Flame Mean (Approach 2: Stats -> Tomo)", -lim_media, lim_media, cmap='RdBu_r')
plot_campo(axs[1, 0], rms_ab1, "Flame Noise-Corrected RMS (Approach 1)", 0, lim_rms, cmap='inferno')
plot_campo(axs[1, 1], rms_ab2, "Flame RMS (Approach 2: Non-physical)", -lim_rms, lim_rms, cmap='RdBu_r')

plt.tight_layout()
caminho_imagem = os.path.join(DIR_PROCESSED, 'comparacao_chama.png')
plt.savefig(caminho_imagem, dpi=300, bbox_inches='tight')
print(f"Image saved at: {caminho_imagem}")
plt.close()