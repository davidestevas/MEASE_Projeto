import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Configure paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')

# 2. Load the ISO case matrices
media_ab1 = np.load(os.path.join(DIR_PROCESSED, 'media_iso_ab1.npy')).copy()
rms_ab1 = np.load(os.path.join(DIR_PROCESSED, 'rms_iso_ab1.npy')).copy()
media_ab2 = np.load(os.path.join(DIR_PROCESSED, 'media_iso_ab2.npy')).copy()
rms_ab2 = np.load(os.path.join(DIR_PROCESSED, 'rms_iso_ab2.npy')).copy()

# 3. Mask the bluff-body region (Z >= 325 px)
Z_MASK = 325
for mat in (media_ab1, rms_ab1, media_ab2, rms_ab2):
    mat[Z_MASK:, :] = 0

fig, axs = plt.subplots(2, 2, figsize=(13, 10))

def plot_campo(ax, matriz, titulo, cmap='RdBu_r', simetrico=True):
    regiao_util = matriz[:Z_MASK, :]
    if simetrico:
        lim = np.percentile(np.abs(regiao_util), 99)
        vmin, vmax = -lim, lim
    else:
        vmin = 0
        vmax = np.percentile(regiao_util, 99)
        
    im = ax.imshow(matriz, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax.axhline(Z_MASK, color='black', linestyle='--', alpha=0.5)
    ax.set_title(titulo, fontsize=12)
    ax.set_xlabel('Radius (px)')
    ax.set_ylabel('Z Height (px)')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Schlieren Signal (a.u.)')

# 4. Draw the 4 plots
plot_campo(axs[0, 0], media_ab1, "Mean ISO Schlieren (Approach 1: Tomo -> Stats)", cmap='RdBu_r', simetrico=True)
plot_campo(axs[0, 1], media_ab2, "Mean ISO Schlieren (Approach 2: Stats -> Tomo)", cmap='RdBu_r', simetrico=True)
plot_campo(axs[1, 0], rms_ab1, "RMS ISO Schlieren (Approach 1)", cmap='cividis', simetrico=False)
plot_campo(axs[1, 1], rms_ab2, "RMS ISO Schlieren (Approach 2: Non-physical)", cmap='RdBu_r', simetrico=True)

plt.tight_layout()

# 5. Save and show
caminho_imagem = os.path.join(DIR_PROCESSED, 'comparacao_iso.png')
plt.savefig(caminho_imagem, dpi=300, bbox_inches='tight')
print(f"Image saved at: {caminho_imagem}")
plt.show()