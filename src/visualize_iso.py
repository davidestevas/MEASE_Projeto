import numpy as np
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')

# Load ISO matrices
media_ab1 = np.load(os.path.join(DIR_PROCESSED, 'media_iso_ab1.npy')).copy()
caminho_iso_corr = os.path.join(DIR_PROCESSED, 'rms_iso_ab1_corrigido.npy')
rms_ab1 = np.load(caminho_iso_corr if os.path.exists(caminho_iso_corr) else os.path.join(DIR_PROCESSED, 'rms_iso_ab1.npy')).copy()
media_ab2 = np.load(os.path.join(DIR_PROCESSED, 'media_iso_ab2.npy')).copy()
rms_ab2 = np.load(os.path.join(DIR_PROCESSED, 'rms_iso_ab2.npy')).copy()

# Load Flame reference matrices to enforce identical color scales (Null-Case presentation)
media_chama_ref = np.load(os.path.join(DIR_PROCESSED, 'media_ab1.npy'))
caminho_chama_corr = os.path.join(DIR_PROCESSED, 'rms_ab1_corrigido.npy')
rms_chama_ref = np.load(caminho_chama_corr if os.path.exists(caminho_chama_corr) else os.path.join(DIR_PROCESSED, 'rms_ab1.npy'))

Z_MASK = 325
for mat in (media_ab1, rms_ab1, media_ab2, rms_ab2):
    mat[Z_MASK:, :] = 0

lim_media_chama = np.percentile(np.abs(media_chama_ref[:Z_MASK, :]), 99)
lim_rms_chama = np.percentile(rms_chama_ref[:Z_MASK, :], 99)

fig, axs = plt.subplots(2, 2, figsize=(13, 10))

def plot_campo(ax, matriz, titulo, vmin, vmax, cmap='RdBu_r'):
    im = ax.imshow(matriz, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax.axhline(Z_MASK, color='black', linestyle='--', alpha=0.5)
    ax.set_title(titulo, fontsize=11.5)
    ax.set_xlabel('Radius r (px) [Same color scale as Flame]')
    ax.set_ylabel('Axial Coordinate z (px)')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'Reconstructed $\partial n / \partial z$ (a.u.)')

plot_campo(axs[0, 0], media_ab1, "ISO Mean (Approach 1 - Null Case, Flame Scale)", -lim_media_chama, lim_media_chama, cmap='RdBu_r')
plot_campo(axs[0, 1], media_ab2, "ISO Mean (Approach 2 - Null Case, Flame Scale)", -lim_media_chama, lim_media_chama, cmap='RdBu_r')
plot_campo(axs[1, 0], rms_ab1, "ISO Noise-Corrected RMS (Approach 1, Flame Scale)", 0, lim_rms_chama, cmap='inferno')
plot_campo(axs[1, 1], rms_ab2, "ISO RMS (Approach 2: Non-physical, Flame Scale)", -lim_rms_chama, lim_rms_chama, cmap='RdBu_r')

plt.tight_layout()
caminho_imagem = os.path.join(DIR_PROCESSED, 'comparacao_iso.png')
plt.savefig(caminho_imagem, dpi=300, bbox_inches='tight')
print(f"Image saved at: {caminho_imagem}")
plt.close()