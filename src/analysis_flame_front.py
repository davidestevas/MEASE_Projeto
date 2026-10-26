import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')

rms_chama = np.load(os.path.join(DIR_PROCESSED, 'rms_ab1.npy'))
rms_iso = np.load(os.path.join(DIR_PROCESSED, 'rms_iso_ab1.npy'))
media_chama = np.load(os.path.join(DIR_PROCESSED, 'media_ab1.npy'))

alturas_z = [40, 80, 120]
RAIO_MIN = 15

fig, axs = plt.subplots(1, 4, figsize=(20, 5))

picos_chama_r = []
picos_chama_z = []

print("--- FLAME FRONT & ISO SHEAR LAYER ---")
for i, z in enumerate(alturas_z):
    raio = np.arange(rms_chama.shape[1])
    perfil_c = rms_chama[z, :]
    perfil_i = rms_iso[z, :]
    
    # 1. Flame: Peak detection in the reaction zone
    picos_c, props_c = find_peaks(perfil_c[RAIO_MIN:], prominence=0.02)
    p_c = None
    if len(picos_c) > 0:
        p_c = picos_c[np.argmax(props_c['prominences'])] + RAIO_MIN
        picos_chama_r.append(p_c)
        picos_chama_z.append(z)
        print(f"Z={z} px | Flame Peak: R={p_c} px")

    # 2. ISO: Radial decay (Half-Width Decay)
    # We assume the ISO reference value right after the axis cut (RAIO_MIN)
    iso_ref_val = np.max(perfil_i[RAIO_MIN:50]) 
    iso_half = iso_ref_val / 2.0
    idx_half = np.where(perfil_i[RAIO_MIN:] < iso_half)[0]
    p_i = idx_half[0] + RAIO_MIN if len(idx_half) > 0 else None
    if p_i: print(f"Z={z} px | ISO Half-Width: R={p_i} px")

    # Plot (Labels in English)
    axs[i].plot(raio, perfil_c, 'r-', label='Flame (RMS)', linewidth=2)
    axs[i].plot(raio, perfil_i, 'b--', label='ISO (RMS)', linewidth=2)
    
    if p_c is not None:
        axs[i].axvline(p_c, color='red', alpha=0.5, linestyle=':', label=f'Flame Front R={p_c}')
    if p_i is not None:
        axs[i].axvline(p_i, color='blue', alpha=0.5, linestyle=':', label=f'ISO Half-Width R={p_i}')
    
    axs[i].set_title(f'Radial Profile Z = {z} px')
    axs[i].set_xlabel('Radius (px)')
    if i == 0: axs[i].set_ylabel('RMS Fluctuation Intensity')
    axs[i].set_xlim(0, 256)
    axs[i].legend()

# 3. Overlay on 2D Map
im = axs[3].imshow(media_chama, cmap='magma', aspect='auto')
axs[3].scatter(picos_chama_r, picos_chama_z, color='cyan', marker='x', s=100, label='Detected Flame Front')
axs[3].set_title('Mean Density Field & Flame Front')
axs[3].set_xlabel('Radius (px)')
axs[3].set_ylabel('Z (px)')
fig.colorbar(im, ax=axs[3], fraction=0.046, pad=0.04)
axs[3].legend()

plt.tight_layout()
caminho_fig = os.path.join(DIR_PROCESSED, 'flame_front_and_iso_decay.png')
plt.savefig(caminho_fig, dpi=300)
print(f"Figure generated for the report: {caminho_fig}")
plt.show()