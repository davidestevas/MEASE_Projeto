import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks

# 1. Configure paths and load unclipped data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')

rms_chama = np.load(os.path.join(DIR_PROCESSED, 'rms_ab1.npy'))
rms_iso = np.load(os.path.join(DIR_PROCESSED, 'rms_iso_ab1.npy'))
media_chama = np.load(os.path.join(DIR_PROCESSED, 'media_ab1.npy')).copy()

alturas_z = [40, 80, 120]
RAIO_MIN = 20   # Ignore 1/r singularity of Onion-Peeling near the axis (R=0)
RAIO_MAX = 150  # Avoid background optical noise at the image edges (R -> 256)

# Mask the bluff-body (Z >= 325 px) to prevent visualization saturation
Z_MASK = 325
media_chama[Z_MASK:, :] = 0

fig, axs = plt.subplots(1, 4, figsize=(20, 5))
picos_chama_r = []
picos_chama_z = []

print("--- FLAME FRONT & ISO SHEAR LAYER (UNCLIPPED) ---")
for i, z in enumerate(alturas_z):
    raio = np.arange(rms_chama.shape[1])
    perfil_c = rms_chama[z, :]
    perfil_i = rms_iso[z, :]
    
    # 1. Flame: Peak detection in the physical window [RAIO_MIN : RAIO_MAX]
    janela_c = perfil_c[RAIO_MIN:RAIO_MAX]
    prominencia_dinamica = 0.10 * np.ptp(janela_c)  # 10% of local peak-to-peak amplitude
    
    picos_c, props_c = find_peaks(janela_c, prominence=prominencia_dinamica, distance=10)
    p_c = None
    if len(picos_c) > 0:
        # Selects the most prominent peak within the physical window
        p_c = picos_c[np.argmax(props_c['prominences'])] + RAIO_MIN
        picos_chama_r.append(p_c)
        picos_chama_z.append(z)
        print(f"Z={z:3d} px | Flame Peak: R={p_c:3d} px (RMS={perfil_c[p_c]:.3f} | Prominence Thr={prominencia_dinamica:.3f})")
    else:
        print(f"Z={z:3d} px | Flame Peak: Not Found")

    # 2. ISO: Radial decay (Half-Width Decay)
    iso_ref_val = np.max(perfil_i[RAIO_MIN:60])
    iso_half = iso_ref_val / 2.0
    idx_half = np.where(perfil_i[RAIO_MIN:] < iso_half)[0]
    p_i = idx_half[0] + RAIO_MIN if len(idx_half) > 0 else None
    if p_i is not None:
        print(f"Z={z:3d} px | ISO Half-Width: R={p_i:3d} px (RMS_half={iso_half:.3f})")

    # Draw the radial profiles
    axs[i].plot(raio, perfil_c, 'r-', label='Flame (RMS)', linewidth=1.8)
    axs[i].plot(raio, perfil_i, 'b--', label='ISO (RMS)', linewidth=1.8)
    
    if p_c is not None:
        axs[i].axvline(p_c, color='red', alpha=0.7, linestyle=':', linewidth=2, label=f'Flame Front R={p_c} px')
    if p_i is not None:
        axs[i].axvline(p_i, color='blue', alpha=0.7, linestyle=':', linewidth=2, label=f'ISO Half-Width R={p_i} px')
    
    axs[i].set_title(f'Radial Profile Z = {z} px')
    axs[i].set_xlabel('Radius (px)')
    if i == 0:
        axs[i].set_ylabel('Reconstructed Schlieren RMS (a.u.)')
    axs[i].set_xlim(0, 200)
    axs[i].grid(True, alpha=0.3)
    axs[i].legend()

# 3. Overlay on the 2D Mean Map (scale adjusted to the 99th percentile of the flame)
limite_cor = np.percentile(np.abs(media_chama[:Z_MASK, :]), 99)
im = axs[3].imshow(media_chama, cmap='RdBu_r', aspect='auto', vmin=-limite_cor, vmax=limite_cor)
axs[3].scatter(picos_chama_r, picos_chama_z, color='lime', edgecolor='black', marker='X', s=120, zorder=5, label='Detected Flame Front')
axs[3].axhline(Z_MASK, color='black', linestyle='--', alpha=0.6, label='Bluff-body Mask')
axs[3].set_title('Reconstructed Mean Schlieren (a.u.)')
axs[3].set_xlabel('Radius (px)')
axs[3].set_ylabel('Z (px)')
fig.colorbar(im, ax=axs[3], fraction=0.046, pad=0.04)
axs[3].legend(loc='lower right')

plt.tight_layout()
caminho_fig = os.path.join(DIR_PROCESSED, 'flame_front_and_iso_decay.png')
plt.savefig(caminho_fig, dpi=300, bbox_inches='tight')
print(f"\nGenerated figure: {caminho_fig}")
plt.show()