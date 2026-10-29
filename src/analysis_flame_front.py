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

caminho_geom = os.path.join(DIR_PROCESSED, 'geometria_queimador.npy')
r_bluff_px = float(np.load(caminho_geom)[1]) if os.path.exists(caminho_geom) else 54.06

Z_MASK = 325
for arr in (media_chama, rms_chama_corr, rms_iso_corr, rms_iso_raw):
    arr[Z_MASK:, :] = 0

RAIO_MIN = 25
RAIO_MAX = 140

def detetar_frente_altura(z_idx):
    """Returns ring indices (j_min_media, j_rms_50). Physical radius is r = j + 0.5 px."""
    perfil_m = uniform_filter1d(media_chama[z_idx, :], size=5)
    perfil_r = uniform_filter1d(rms_chama_corr[z_idx, :], size=5)
    
    j_min_media = int(np.argmin(perfil_m[RAIO_MIN:RAIO_MAX]) + RAIO_MIN)
    
    rms_patamar = np.max(perfil_r[RAIO_MIN:110])
    indices_acima_50 = np.where(perfil_r[RAIO_MIN:160] >= 0.50 * rms_patamar)[0]
    j_rms_50 = int(indices_acima_50[-1] + RAIO_MIN) if len(indices_acima_50) > 0 else j_min_media
    return j_min_media, j_rms_50

z_todas = np.arange(20, 316)
r_frente_media = []
r_frente_rms50 = []

for z_val in z_todas:
    jm, jr = detetar_frente_altura(z_val)
    r_frente_media.append(jm + 0.5)
    r_frente_rms50.append(jr + 0.5)

r_frente_media_smooth = uniform_filter1d(np.array(r_frente_media, dtype=np.float64), size=9)
r_frente_rms50_smooth = uniform_filter1d(np.array(r_frente_rms50, dtype=np.float64), size=9)

print("--- FLAME FRONT TRACKING (PHYSICAL RADIUS r = j + 0.5 px AND DIMENSIONLESS r / R_b) ---")
rm_dev, rr_dev = [], []
for z_check in [40, 60, 80, 100, 120, 160, 200]:
    jm, jr = detetar_frente_altura(z_check)
    rm_phys, rr_phys = jm + 0.5, jr + 0.5
    rm_dev.append(rm_phys)
    rr_dev.append(rr_phys)
    iso_u_half = np.where(rms_iso_raw[z_check, 20:] < 0.5 * np.max(rms_iso_raw[z_check, 20:60]))[0]
    iso_c_half = np.where(rms_iso_corr[z_check, 20:] < 0.5 * np.max(rms_iso_corr[z_check, 20:60]))[0]
    r_iso_u = (int(iso_u_half[0] + 20) if len(iso_u_half) > 0 else 20) + 0.5
    r_iso_c = (int(iso_c_half[0] + 20) if len(iso_c_half) > 0 else 20) + 0.5
    print(f"Z={z_check:3d} px | Band Center: j={jm:2d} (r={rm_phys:5.1f} px, r/R_b={rm_phys/r_bluff_px:.2f}) | "
          f"Outer 50% RMS: j={jr:3d} (r={rr_phys:5.1f} px, r/R_b={rr_phys/r_bluff_px:.2f}) | "
          f"ISO Half-Width: {r_iso_u:4.1f}->{r_iso_c:4.1f} px")

alturas_z = [40, 80, 120]
fig, axs = plt.subplots(1, 4, figsize=(21, 5.2))

for i, z in enumerate(alturas_z):
    raio = np.arange(rms_chama_corr.shape[1]) + 0.5
    perfil_m = media_chama[z, :]
    perfil_c = rms_chama_corr[z, :]
    perfil_i_corr = rms_iso_corr[z, :]
    perfil_i_raw = rms_iso_raw[z, :]
    
    jm, jr50 = detetar_frente_altura(z)
    rm_p, rr_p = jm + 0.5, jr50 + 0.5
    
    axs[i].plot(raio, perfil_c, 'r-', linewidth=1.8, label='Flame Corrected RMS')
    axs[i].plot(raio, -perfil_m, 'k-', alpha=0.75, linewidth=1.5, label=r'$-\mathrm{Mean}\ (\partial n/\partial z)$')
    axs[i].plot(raio, perfil_i_raw, 'b:', alpha=0.6, linewidth=1.2, label='ISO Uncorrected RMS')
    axs[i].plot(raio, perfil_i_corr, 'b-', linewidth=1.4, label='ISO Corrected RMS (Null)')
    
    axs[i].axvline(rm_p, color='green', linestyle='-', linewidth=2.0,
                   label=f'Band Center r={rm_p:.1f} px ({rm_p/r_bluff_px:.2f}$R_b$)')
    axs[i].axvline(rr_p, color='darkorange', linestyle='--', linewidth=2.0,
                   label=f'Outer 50% RMS r={rr_p:.1f} px ({rr_p/r_bluff_px:.2f}$R_b$)')
    
    axs[i].set_title(f'Radial Profiles at Z = {z} px')
    axs[i].set_xlabel('Physical Radius r = j + 0.5 (px)')
    if i == 0:
        axs[i].set_ylabel(r'Reconstructed $\partial n / \partial z$ Signal (a.u.)')
    axs[i].set_xlim(0, 180)
    axs[i].grid(True, alpha=0.3)
    axs[i].legend(fontsize=7.8, loc='center right')

limite_cor = np.percentile(np.abs(media_chama[:Z_MASK, :]), 99)
im = axs[3].imshow(media_chama, cmap='RdBu_r', aspect='auto', vmin=-limite_cor, vmax=limite_cor)
axs[3].plot(r_frente_media_smooth, z_todas, color='lime', linewidth=2.4,
            label=f'Band Center: r={min(rm_dev):.1f}–{max(rm_dev):.1f} px ({min(rm_dev)/r_bluff_px:.2f}–{max(rm_dev)/r_bluff_px:.2f}$R_b$)')
axs[3].plot(r_frente_rms50_smooth, z_todas, color='gold', linestyle='--', linewidth=2.2,
            label=f'Outer 50% RMS: r={min(rr_dev):.1f}–{max(rr_dev):.1f} px ({min(rr_dev)/r_bluff_px:.2f}–{max(rr_dev)/r_bluff_px:.2f}$R_b$)')
axs[3].axhline(Z_MASK, color='black', linestyle='--', alpha=0.6, label='Bluff-body Mask (Z=325)')
axs[3].scatter([r_bluff_px], [320], color='cyan', edgecolor='black', marker='^', s=100, zorder=6,
               label=f'Bluff-body Lip (R_b={r_bluff_px:.1f} px)')

axs[3].set_title(r'Mean $\partial n / \partial z$ (a.u.) & Operational Boundaries')
axs[3].set_xlabel('Physical Radius r = j + 0.5 (px)')
axs[3].set_ylabel('Axial Coordinate z (px)')
fig.colorbar(im, ax=axs[3], fraction=0.046, pad=0.04)
axs[3].legend(loc='lower right', fontsize=7.5)

plt.tight_layout()
caminho_fig = os.path.join(DIR_PROCESSED, 'flame_front_and_iso_decay.png')
plt.savefig(caminho_fig, dpi=300, bbox_inches='tight')
print(f"\nGenerated figure: {caminho_fig}")
plt.close()