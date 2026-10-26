import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Configure paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')

# 2. Load the ISO case matrices saved by approach2.py
media_ab1 = np.load(os.path.join(DIR_PROCESSED, 'media_iso_ab1.npy'))
rms_ab1 = np.load(os.path.join(DIR_PROCESSED, 'rms_iso_ab1.npy'))
media_ab2 = np.load(os.path.join(DIR_PROCESSED, 'media_iso_ab2.npy'))
rms_ab2 = np.load(os.path.join(DIR_PROCESSED, 'rms_iso_ab2.npy'))

# 3. Prepare the figure for comparison (2x2)
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

def plot_campo(ax, matriz, titulo, cmap='viridis'):
    # For the ISO case (without combustion), 'viridis' (green/blue/yellow tones) 
    # helps to visually distinguish from the Flame ('magma' - fire tones)
    im = ax.imshow(matriz, cmap=cmap) 
    ax.set_title(titulo, fontsize=14)
    ax.set_xlabel('Radius (px)')
    ax.set_ylabel('Z Height (px)')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# 4. Draw the 4 plots
plot_campo(axs[0, 0], media_ab1, "Mean ISO (Approach 1: Tomo -> Stats)")
plot_campo(axs[0, 1], media_ab2, "Mean ISO (Approach 2: Stats -> Tomo)")
plot_campo(axs[1, 0], rms_ab1, "RMS ISO (Approach 1)", cmap='cividis')
plot_campo(axs[1, 1], rms_ab2, "RMS ISO (Approach 2)", cmap='cividis')

plt.tight_layout()

# 5. Show on screen and save the image
caminho_imagem = os.path.join(DIR_PROCESSED, 'comparacao_iso.png')
plt.savefig(caminho_imagem, dpi=300, bbox_inches='tight')
print(f"Image saved at: {caminho_imagem}")

plt.show()