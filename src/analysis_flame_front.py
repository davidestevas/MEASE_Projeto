import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import uniform_filter1d

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')

media_chama = np.load(os.path.join(DIR_PROCESSED, 'media_ab1.npy')).copy()
rms_chama_corr = np.load(os.path.join(DIR_PROCESSED, 'rms_ab1_corrigido.npy')).copy()
rms_iso_corr = np.load(os.path.join(DIR_PROCESSED, 'rms_iso_ab1_corrigido.npy')).copy()

Z_MASK = 325
media_chama[Z_MASK:, :] = 0
rms_chama_corr[Z_MASK:, :] = 0
rms_iso_corr[Z_MASK:, :] = 0

RAIO_MIN = 25
RAIO_MAX = 140

def detetar_frente_altura(z_idx):
    """
    Returns:
      r_min_media: Primary criterion (minimum of the mean axial gradient band)
      r_bordo_rms: Verification criterion (outer peak/edge of the corrected RMS plateau)
    """
    perfil_m = uniform_filter1d(media_chama[z_idx, :], size=5)
    perfil_r = uniform_filter1d(rms_chama_corr[z_idx, :], size=5)
    
    # 1. Primary Criterion: Minimum of Mean Field in [RAIO_MIN, RAIO_MAX]
    r_min_media = int(np.argmin(perfil_m[RAIO_MIN:RAIO_MAX]) + RAIO_MIN)
    
    # 2. Verification Criterion: Outer edge/peak of the RMS plateau (search around outer shear layer)
    r_busca_ini = max(RAIO_MIN, r_min_media - 15)
    r_busca_fim = min(RAIO_MAX, r_min_media + 25)
    r_bordo_rms = int(np.argmax(perfil_r[r_busca_ini:r_busca_fim]) + r_busca_ini)
    return r_min_media, r_bordo_rms

# Trace flame front across ALL axial heights (from bluff-body lip z=315 up to z=20 px)
z_todas = np.arange(20, 316)
r_frente_media = []
r_frente_rms = []

for z_val in z_todas:
    rm, rr = detetar_frente_altura(z_val)
    r_frente_media.append(rm)
    r_frente_rms.append(rr)

r_frente_media = uniform_filter1d(np.array(r_frente_media, dtype=np.float64), size=9)
r_frente_rms = uniform_filter1d(np.array(r_frente_rms, dtype=np.float64), size=9)

alturas_z = [40, 80, 120]
fig, axs = plt.subplots(1, 4, figsize=(21, 5.2))

print("--- FLAME FRONT TRACKING (PRIMARY: MEAN MINIMUM | VERIFICATION: CORRECTED RMS OUTER EDGE) ---")
for i, z in enumerate(alturas_z):
    raio = np.arange(rms_chama_corr.shape[1])
    perfil_m = media_chama[z, :]
    perfil_c = rms_chama_corr[z, :]
    perfil_i = rms_iso_corr[z, :]
    
    r_m, r_r = detetar_frente_altura(z)
    print(f"Z={z:3d} px | Mean Min (Primary): R={r_m:3d} px (Mean={perfil_m[r_m]:+.3f}) | "
          f"Corr. RMS Outer Edge (Verif.): R={r_r:3d} px (RMS_corr={perfil_c[r_r]:.3f})")
    
    axs[i].plot(raio, perfil_c, 'r-', linewidth=1.8, label='Flame Corrected RMS')
    axs[i].plot(raio, -perfil_m, 'k-', alpha=0.75, linewidth=1.5, label=r'$-\mathrm{Mean}\ (\partial n/\partial z)$')
    axs[i].plot(raio, perfil_i, 'b--', linewidth=1.4, label='ISO Corrected RMS (Null)')
    
    axs[i].axvline(r_m, color='green', linestyle='-', linewidth=2.0, label=f'Mean Min R={r_m} px')
    axs[i].axvline(r_r, color='darkred', linestyle=':', linewidth=2.0, label=f'RMS Outer R={r_r} px')
    
    axs[i].set_title(f'Radial Profiles at Z = {z} px')
    axs[i].set_xlabel('Radius r (px)')
    if i == 0:
        axs[i].set_ylabel(r'Reconstructed $\partial n / \partial z$ Signal (a.u.)')
    axs[i].set_xlim(0, 180)
    axs[i].grid(True, alpha=0.3)
    axs[i].legend(fontsize=8.5, loc='upper right')

print(f"\nFull Axial Evolution Summary:")
print(f"   -> Cylinder region (Z=40..200 px): Mean Min R = {np.min(r_frente_media[20:180]):.0f} to {np.max(r_frente_media[20:180]):.0f} px")
print(f"   -> Bluff-body anchoring (Z=300..315 px): Mean Min R = {np.mean(r_frente_media[-15:]):.1f} px (Bluff-body lip R_b ≈ 58 px)")

# 4th subplot: Full 2D Mean Map with complete axial flame front trajectory
limite_cor = np.percentile(np.abs(media_chama[:Z_MASK, :]), 99)
im = axs[3].imshow(media_chama, cmap='RdBu_r', aspect='auto', vmin=-limite_cor, vmax=limite_cor)
axs[3].plot(r_frente_media, z_todas, color='lime', linewidth=2.5, label='Primary: Mean Minimum (z=20..315)')
axs[3].plot(r_frente_rms, z_todas, color='yellow', linestyle='--', linewidth=1.8, label='Verif.: Corr. RMS Outer Edge')
axs[3].axhline(Z_MASK, color='black', linestyle='--', alpha=0.6, label='Bluff-body Mask (Z=325)')
axs[3].scatter([58], [320], color='cyan', edgecolor='black', marker='^', s=100, zorder=6, label='Bluff-body Lip (R≈58 px)')

axs[3].set_title(r'Mean $\partial n / \partial z$ (a.u.) & Full Flame Front')
axs[3].set_xlabel('Radius r (px)')
axs[3].set_ylabel('Axial Coordinate z (px)')
fig.colorbar(im, ax=axs[3], fraction=0.046, pad=0.04)
axs[3].legend(loc='lower right', fontsize=8)

plt.tight_layout()
caminho_fig = os.path.join(DIR_PROCESSED, 'flame_front_and_iso_decay.png')
plt.savefig(caminho_fig, dpi=300, bbox_inches='tight')
print(f"Generated figure: {caminho_fig}")
plt.close()