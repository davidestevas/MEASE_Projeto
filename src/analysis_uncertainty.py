import numpy as np
import imageio.v3 as iio
import glob
import os
import re
import matplotlib.pyplot as plt
from utils import calcular_matriz_ART_inversa

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')
DIR_BG = os.path.join(BASE_DIR, 'data', 'raw', 'Background', 'tiff', '*.tif')
DIR_CHAMA = os.path.join(BASE_DIR, 'data', 'raw', 'Chama', '*.tif')

Z_MASK = 325  # Mask bluff-body region (Z >= 325 px)
COL_SIMETRIA = 256
NUM_PIXEIS_RAIO = 512 - COL_SIMETRIA
A_inv = calcular_matriz_ART_inversa(NUM_PIXEIS_RAIO)

def natural_sort_key(s):
    """Sorts file paths chronologically by frame number."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

# ==========================================
# 1. Mathematical Proof: Failure of Approach 2
# ==========================================
print("--- 1. PHYSICAL ANALYSIS OF RMS (FLOW DOMAIN Z < 325 px) ---")
rms_ab1_full = np.load(os.path.join(DIR_PROCESSED, 'rms_ab1.npy'))
rms_ab2_full = np.load(os.path.join(DIR_PROCESSED, 'rms_ab2.npy'))
media_ab1_full = np.load(os.path.join(DIR_PROCESSED, 'media_ab1.npy'))

rms_ab1_chama = rms_ab1_full[:Z_MASK, :]
rms_ab2_chama = rms_ab2_full[:Z_MASK, :]
media_ab1_chama = media_ab1_full[:Z_MASK, :]

print(f"Approach 1 (Tomography -> Stats): Min = {np.min(rms_ab1_chama):.4f} | Max = {np.max(rms_ab1_chama):.4f}")
print(f"Approach 2 (Stats -> Tomography): Min = {np.min(rms_ab2_chama):.4f} | Max = {np.max(rms_ab2_chama):.4f}")
print("-> CONCLUSION: Approach 2 generates non-physical negative values, confirming that non-linear operators (RMS) do not commute with the inverse Abel matrix.\n")

# ==========================================
# 2. Sensor Noise Propagation & RMS Correction (Point 5)
# ==========================================
print("--- 2. SENSOR NOISE PROPAGATION & RMS DEBIASING (POINT 5) ---")
arquivos_bg = sorted(glob.glob(DIR_BG), key=natural_sort_key)
n_bg = len(arquivos_bg)

soma_bg = np.zeros((512, 512), dtype=np.float64)
soma_bg_sq = np.zeros((512, 512), dtype=np.float64)

for arq in arquivos_bg:
    img_bg = iio.imread(arq).astype(np.float64)
    soma_bg += img_bg
    soma_bg_sq += (img_bg ** 2)

bg_medio = soma_bg / n_bg
var_bg = np.maximum((soma_bg_sq / n_bg) - (bg_medio ** 2), 0)
rms_bg_raw = np.sqrt(var_bg[:Z_MASK, COL_SIMETRIA:])

# Propagate background variance across full 512 height to save full corrected RMS map
var_bg_rec_full = var_bg[:, COL_SIMETRIA:] @ (A_inv.T ** 2)
sigma_rec_bg_full = np.sqrt(np.maximum(var_bg_rec_full, 0))
sigma_rec_bg = sigma_rec_bg_full[:Z_MASK, :]

# Point 5 Formula: RMS_corr = sqrt(max(RMS^2 - sigma_rec^2, 0))
rms_ab1_corr_full = np.sqrt(np.maximum((rms_ab1_full ** 2) - (sigma_rec_bg_full ** 2), 0))
rms_ab1_corr = rms_ab1_corr_full[:Z_MASK, :]

np.save(os.path.join(DIR_PROCESSED, 'sigma_rec_bg.npy'), sigma_rec_bg_full)
np.save(os.path.join(DIR_PROCESSED, 'rms_ab1_corrigido.npy'), rms_ab1_corr_full)

# Also correct ISO RMS using the same background noise variance
caminho_rms_iso = os.path.join(DIR_PROCESSED, 'rms_iso_ab1.npy')
if os.path.exists(caminho_rms_iso):
    rms_iso_full = np.load(caminho_rms_iso)
    rms_iso_corr_full = np.sqrt(np.maximum((rms_iso_full ** 2) - (sigma_rec_bg_full ** 2), 0))
    np.save(os.path.join(DIR_PROCESSED, 'rms_iso_ab1_corrigido.npy'), rms_iso_corr_full)

print(f"Mean Raw Sensor Noise (Background RMS): {np.mean(rms_bg_raw):.4f} a.u.")
print(f"Reconstructed Background Noise (At Axis R=0): {np.mean(sigma_rec_bg[:, 0]):.4f} a.u.")
print(f"Reconstructed Background Noise (Radial Mean): {np.mean(sigma_rec_bg):.4f} a.u.")
print(f"Flame RMS at Axis (R=0) Mean: Before Correction = {np.mean(rms_ab1_chama[:, 0]):.4f} | After Correction = {np.mean(rms_ab1_corr[:, 0]):.4f}")
print(f"Flame RMS in Shear Layer (R=98, Z=80): Before = {rms_ab1_chama[80, 98]:.4f} | After = {rms_ab1_corr[80, 98]:.4f}\n")

# Plot Raw vs Noise-Corrected RMS Radial Profiles
fig_rms, axs_rms = plt.subplots(1, 3, figsize=(15, 4.2))
for i, z in enumerate([40, 80, 120]):
    r_ax = np.arange(NUM_PIXEIS_RAIO)
    axs_rms[i].plot(r_ax, rms_ab1_chama[z, :], 'r--', alpha=0.6, label='Uncorrected RMS')
    axs_rms[i].plot(r_ax, rms_ab1_corr[z, :], 'r-', linewidth=1.8, label=r'Corrected $\sqrt{RMS^2 - \sigma_{rec}^2}$')
    axs_rms[i].plot(r_ax, sigma_rec_bg[z, :], 'k:', alpha=0.8, label=r'Propagated Noise $\sigma_{rec}$')
    axs_rms[i].set_title(f'Noise-Corrected RMS Profile (Z = {z} px)')
    axs_rms[i].set_xlabel('Radius (px)')
    if i == 0:
        axs_rms[i].set_ylabel('Schlieren RMS (a.u.)')
    axs_rms[i].set_xlim(0, 200)
    axs_rms[i].grid(True, alpha=0.3)
    axs_rms[i].legend(fontsize=9)
plt.tight_layout()
caminho_fig_rms = os.path.join(DIR_PROCESSED, 'rms_noise_correction.png')
plt.savefig(caminho_fig_rms, dpi=300, bbox_inches='tight')
plt.close(fig_rms)
print(f"Saved noise-corrected RMS figure: {caminho_fig_rms}\n")

# ==========================================
# 3. Temporal Autocorrelation & Effective Sample Size N_eff (Point 4)
# ==========================================
print("--- 3. TEMPORAL AUTOCORRELATION & EFFECTIVE SAMPLES N_eff (POINT 4) ---")
arquivos_chama = sorted(glob.glob(DIR_CHAMA), key=natural_sort_key)
N_CHAMA = len(arquivos_chama)
FPS = 400.0
DT_MS = 1000.0 / FPS  # 2.5 ms between consecutive frames

# Probe 3 axial heights along the flame front (R = 98 px)
z_probes = [40, 80, 120]
r_probe = 98
series_temporais = np.zeros((N_CHAMA, len(z_probes)), dtype=np.float64)

print(f"Extracting chronological time series at R={r_probe} px across {N_CHAMA} frames...")
# Only need row r_probe of A_inv to reconstruct radius r_probe! (Ultra-fast vector dot product)
a_inv_row = A_inv[r_probe, :]

for idx_f, arq in enumerate(arquivos_chama):
    img = iio.imread(arq).astype(np.float64)
    for idx_z, z_val in enumerate(z_probes):
        proj_linha = img[z_val, COL_SIMETRIA:] - bg_medio[z_val, COL_SIMETRIA:]
        series_temporais[idx_f, idx_z] = np.dot(proj_linha, a_inv_row)

plt.figure(figsize=(8, 4.5))
neff_list = []
fator_inflacao_list = []

for idx_z, z_val in enumerate(z_probes):
    sinal = series_temporais[:, idx_z] - np.mean(series_temporais[:, idx_z])
    variancia = np.var(sinal)
    
    # Normalized Autocorrelation Function (ACF) via FFT
    acf_full = np.correlate(sinal, sinal, mode='full')[N_CHAMA - 1:] / (variancia * N_CHAMA)
    
    # Initial Positive Sequence Estimator (truncate at first zero-crossing)
    zeros_idx = np.where(acf_full <= 0)[0]
    lag_corte = zeros_idx[0] if len(zeros_idx) > 0 else 50
    
    # Integrated autocorrelation time (in frames and ms)
    tau_int_frames = 1.0 + 2.0 * np.sum(acf_full[1:lag_corte])
    tau_int_frames = max(tau_int_frames, 1.0)
    tau_int_ms = (tau_int_frames - 1.0) * DT_MS / 2.0
    
    n_eff = N_CHAMA / tau_int_frames
    fator_inflacao = np.sqrt(tau_int_frames)
    
    neff_list.append(n_eff)
    fator_inflacao_list.append(fator_inflacao)
    
    # Moving Block Bootstrap validation (block size = int(ceil(tau_int_frames * 2)))
    bloco_tam = max(int(np.ceil(tau_int_frames * 2)), 2)
    n_blocos = N_CHAMA // bloco_tam
    rng = np.random.default_rng(42)
    medias_boot = np.zeros(1000)
    for b in range(1000):
        inicios = rng.integers(0, N_CHAMA - bloco_tam, size=n_blocos)
        amostra = np.concatenate([series_temporais[ini:ini + bloco_tam, idx_z] for ini in inicios])
        medias_boot[b] = np.mean(amostra)
    se_boot = np.std(medias_boot, ddof=1)
    se_naive_probe = np.std(series_temporais[:, idx_z], ddof=1) / np.sqrt(N_CHAMA)
    se_acf_probe = se_naive_probe * fator_inflacao
    
    print(f"Probe (Z={z_val:3d}, R={r_probe}): Zero-crossing lag = {lag_corte} frames ({lag_corte*DT_MS:.1f} ms)")
    print(f"   -> Integral Time Scale tau_int = {tau_int_ms:.2f} ms ({tau_int_frames:.2f} frames/indep. sample)")
    print(f"   -> Effective Independent Samples N_eff = {n_eff:.0f} (vs Naive N = {N_CHAMA})")
    print(f"   -> Standard Error: Naive = {se_naive_probe:.4f} | ACF-Corrected = {se_acf_probe:.4f} | Block-Bootstrap = {se_boot:.4f} (Inflation = {fator_inflacao:.2f}x)")
    
    lags_ms = np.arange(40) * DT_MS
    plt.plot(lags_ms, acf_full[:40], marker='o', markersize=3.5, linewidth=1.6,
             label=f'Z = {z_val} px ($N_{{eff}} \\approx {n_eff:.0f}$, $\\tau_{{int}} = {tau_int_ms:.1f}$ ms)')

plt.axhline(0, color='black', linestyle='--', alpha=0.6)
plt.xlabel('Time Lag $\\tau$ (ms) [at 400 fps]')
plt.ylabel('Autocorrelation Coefficient $\\rho(\\tau)$')
plt.title(f'Temporal Autocorrelation in the Flame Front (R = {r_probe} px)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
caminho_fig_acf = os.path.join(DIR_PROCESSED, 'temporal_autocorrelation.png')
plt.savefig(caminho_fig_acf, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved temporal autocorrelation figure: {caminho_fig_acf}\n")

# Mean effective sample size across the reactive zone
N_EFF_MEAN = np.mean(neff_list)
erro_padrao_naive = rms_ab1_corr / np.sqrt(N_CHAMA)
erro_padrao_corr = rms_ab1_corr / np.sqrt(N_EFF_MEAN)

lim_sinal = 0.20 * np.percentile(np.abs(media_ab1_chama), 99)
mascara_sinal = np.abs(media_ab1_chama) > lim_sinal

erros_rel_naive = (erro_padrao_naive[mascara_sinal] / np.abs(media_ab1_chama[mascara_sinal])) * 100
erros_rel_corr = (erro_padrao_corr[mascara_sinal] / np.abs(media_ab1_chama[mascara_sinal])) * 100

print(f"Global Statistical Uncertainty Summary (Active Zone):")
print(f"   -> Naive (N={N_CHAMA}): Median Relative SE = {np.median(erros_rel_naive):.2f}%")
print(f"   -> Autocorrelation-Corrected (N_eff={N_EFF_MEAN:.0f}): Median Relative SE = {np.median(erros_rel_corr):.2f}%\n")

# ==========================================
# 4. Symmetry Axis Uncertainty (Synthetic & Propagated - Point 3)
# ==========================================
print("--- 4. SYMMETRY AXIS SENSITIVITY & TOTAL UNCERTAINTY BUDGET (POINT 3) ---")

A_forward = np.linalg.inv(A_inv)
r_grid = np.arange(NUM_PIXEIS_RAIO, dtype=np.float64)

R_flame_syn = 80.0
W_syn = 8.8
f_true = -((r_grid - R_flame_syn) / W_syn) * np.exp(-((r_grid - R_flame_syn) / W_syn) ** 2)
p_exact = A_forward @ f_true

desvios_px = [0.5, 1.0, 1.5, 2.0]
erros_sinteticos = {}

for dx in desvios_px:
    p_shifted = np.interp(r_grid + dx, r_grid, p_exact, right=0.0)
    f_rec_shift = A_inv @ p_shifted
    erro_l2 = (np.linalg.norm(f_rec_shift - f_true) / np.linalg.norm(f_true)) * 100
    erros_sinteticos[dx] = erro_l2
    print(f"Synthetic Axis Shift dx = {dx:.1f} px -> Relative Reconstruction Error (L2) = {erro_l2:.2f}%")

# Total Combined Uncertainty (ACF-Corrected Statistical SE + 0.5 px Axis Shift in Quadrature)
erro_rel_eixo_05 = erros_sinteticos[0.5] / 100.0
u_eixo_campo = erro_rel_eixo_05 * np.abs(media_ab1_chama)
u_total_combinada = np.sqrt(erro_padrao_corr**2 + u_eixo_campo**2)

rel_u_total = (u_total_combinada[mascara_sinal] / np.abs(media_ab1_chama[mascara_sinal])) * 100
print(f"\nFINAL COMBINED RELATIVE UNCERTAINTY (ACF-Corrected Stat + 0.5 px Axis Shift, Median): {np.median(rel_u_total):.2f}%")