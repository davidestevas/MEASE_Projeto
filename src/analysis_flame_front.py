import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import uniform_filter1d

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')

media_chama = np.load(os.path.join(DIR_PROCESSED, 'media_ab1.npy')).copy()
rms_chama_corr = np.load(os.path.join(DIR_PROCESSED, 'rms_ab1_corrigido.npy')).copy()
rms_iso_corr = np.load(os.path.join(DIR_PROCESSED, 'rms_iso_ab1_corrigido.npy')).copy()
rms_iso_raw = np.load(os.path.join(DIR_PROCESSED, 'rms_iso_ab1.npy')).copy()

Z_MASK = 325
for arr in (media_chama, rms_chama_corr, rms_iso_corr, rms_iso_raw):
    arr[Z_MASK:, :] = 0

RAIO_MIN = 25
RAIO_MAX = 140

def detetar_frente_altura(z_idx):
    """
    Operational definitions of the flame front at axial coordinate z_idx:
      1. r_min_media (Primary): Minimum of the mean axial gradient band (dn/dz extremum, r ≈ 78-85 px).
      2. r_rms_50 (Independent Verification): Outer radius where the noise-corrected RMS drops
         to 50% of its shear-layer plateau maximum (r ≈ 101-108 px).
    """
    perfil_m = uniform_filter1d(media_chama[z_idx, :], size=5)
    perfil_r = uniform_filter1d(rms_chama_corr[z_idx, :], size=5)
    
    # 1. Primary Criterion: Minimum of Mean Field in [RAIO_MIN, RAIO_MAX]
    r_min_media = int(np.argmin(perfil_m[RAIO_MIN:RAIO_MAX]) + RAIO_MIN)
    
    # 2. Independent Verification Criterion: Outer radius where corrected RMS drops to 50% of plateau
    rms_patamar = np.max(perfil_r[RAIO_MIN:110])
    indices_acima_50 = np.where(perfil_r[RAIO_MIN:160] >= 0.50 * rms_patamar)[0]
    if len(indices_acima_50) > 0:
        r_rms_50 = int(indices_acima_50[-1] + RAIO_MIN)
    else:
        r_rms_50 = r_min_media
    return r_min_media, r_rms_50

z_todas = np.arange(20, 316)
r_frente_media = []
r_frente_rms50 = []

for z_val in z_todas:
    rm, rr = detetar_frente_altura(z_val)
    r_frente_media.append(rm)
    r_frente_rms50.append(rr)

r_frente_media_smooth = uniform_filter1d(np.array(r_frente_media, dtype=np.float64), size=9)
r_frente_rms50_smooth = uniform_filter1d(np.array(r_frente_rms50, dtype=np.float64), size=9)

print("--- FLAME FRONT TRACKING (CENTER OF BAND vs OUTER 50% RMS BOUNDARY) ---")
for z_check in [40, 60, 80, 100, 120, 160, 200]:
    rm, rr = detetar_frente_altura(z_check)
    # Also compute ISO half-width collapse (Uncorrected vs Corrected)
    iso_u_half = np.where(rms_iso_raw[z_check, 20:] < 0.5 * np.max(rms_iso_raw[z_check, 20:60]))[0]
    iso_c_half = np.where(rms_iso_corr[z_check, 20:] < 0.5 * np.max(rms_iso_corr[z_check, 20:60]))[0]
    r_iso_u = int(iso_u_half[0] + 20) if len(iso_u_half) > 0 else 20
    r_iso_c = int(iso_c_half[0] + 20) if len(iso_c_half) > 0 else 20
    print(f"Z={z_check:3d} px | Band Center (Mean Min): R={rm:3d} px | "
          f"Outer 50% Corr. RMS: R={rr:3d} px | ISO Half-Width (Uncorr->Corr): {r_iso_u:2d}->{r_iso_c:2d} px")

alturas_z = [40, 80, 120]
fig, axs = plt.subplots(1, 4, figsize=(21, 5.2))

for i, z in enumerate(alturas_z):
    raio = np.arange(rms_chama_corr.shape[1])
    perfil_m = media_chama[z, :]
    perfil_c = rms_chama_corr[z, :]
    perfil_i_corr = rms_iso_corr[z, :]
    perfil_i_raw = rms_iso_raw[z, :]
    
    r_m, r_r50 = detetar_frente_altura(z)
    
    axs[i].plot(raio, perfil_c, 'r-', linewidth=1.8, label='Flame Corrected RMS')
    axs[i].plot(raio, -perfil_m, 'k-', alpha=0.75, linewidth=1.5, label=r'$-\mathrm{Mean}\ (\partial n/\partial z)$')
    axs[i].plot(raio, perfil_i_raw, 'b:', alpha=0.6, linewidth=1.2, label='ISO Uncorrected RMS')
    axs[i].plot(raio, perfil_i_corr, 'b-', linewidth=1.4, label='ISO Corrected RMS (Null)')
    
    axs[i].axvline(r_m, color='green', linestyle='-', linewidth=2.0, label=f'Band Center R={r_m} px')
    axs[i].axvline(r_r50, color='darkorange', linestyle='--', linewidth=2.0, label=f'Outer 50% RMS R={r_r50} px')
    
    axs[i].set_title(f'Radial Profiles at Z = {z} px')
    axs[i].set_xlabel('Radius r (px)')
    if i == 0:
        axs[i].set_ylabel(r'Reconstructed $\partial n / \partial z$ Signal (a.u.)')
    axs[i].set_xlim(0, 180)
    axs[i].grid(True, alpha=0.3)
    axs[i].legend(fontsize=8, loc='upper right')

limite_cor = np.percentile(np.abs(media_chama[:Z_MASK, :]), 99)
im = axs[3].imshow(media_chama, cmap='RdBu_r', aspect='auto', vmin=-limite_cor, vmax=limite_cor)
axs[3].plot(r_frente_media_smooth, z_todas, color='lime', linewidth=2.4, label='Band Center: Mean Min (78–85 px)')
axs[3].plot(r_frente_rms50_smooth, z_todas, color='gold', linestyle='--', linewidth=2.2, label='Outer 50% Corr. RMS (~107 px)')
axs[3].axhline(Z_MASK, color='black', linestyle='--', alpha=0.6, label='Bluff-body Mask (Z=325)')
axs[3].scatter([56], [320], color='cyan', edgecolor='black', marker='^', s=100, zorder=6, label='Bluff-body Lip (R_b≈56 px)')

axs[3].set_title(r'Mean $\partial n / \partial z$ (a.u.) & Operational Boundaries')
axs[3].set_xlabel('Radius r (px)')
axs[3].set_ylabel('Axial Coordinate z (px)')
fig.colorbar(im, ax=axs[3], fraction=0.046, pad=0.04)
axs[3].legend(loc='lower right', fontsize=8)

plt.tight_layout()
caminho_fig = os.path.join(DIR_PROCESSED, 'flame_front_and_iso_decay.png')
plt.savefig(caminho_fig, dpi=300, bbox_inches='tight')
print(f"\nGenerated figure: {caminho_fig}")
plt.close()