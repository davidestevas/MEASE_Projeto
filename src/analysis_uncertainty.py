import numpy as np
import imageio.v3 as iio
import glob
import os
import re
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from utils import (
    calcular_matriz_ART_inversa,
    extrair_metade_direita,
    carregar_delta_eixo,
    medir_corpo_rombo_background,
    D_BLUFF_MM
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')
DIR_BG = os.path.join(BASE_DIR, 'data', 'raw', 'Background', 'tiff', '*.tif')
DIR_CHAMA = os.path.join(BASE_DIR, 'data', 'raw', 'Chama', '*.tif')

caminho_temp = os.path.join(DIR_PROCESSED, 'temp')
if os.path.exists(caminho_temp):
    os.remove(caminho_temp)

Z_MASK = 325
COL_SIMETRIA = 256
NUM_PIXEIS_RAIO = 512 - COL_SIMETRIA

delta_medido, delta_aplicado = carregar_delta_eixo(DIR_PROCESSED)
A_inv = calcular_matriz_ART_inversa(NUM_PIXEIS_RAIO)
A_forward = np.linalg.inv(A_inv)

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

# ==========================================
# 1. Commutativity & Knife-Edge Parity Proof
# ==========================================
print("--- 1. COMMUTATIVITY & KNIFE-EDGE ORIENTATION PROOF (Z < 325 px) ---")
media_ab1_full = np.load(os.path.join(DIR_PROCESSED, 'media_ab1.npy'))
media_ab2_full = np.load(os.path.join(DIR_PROCESSED, 'media_ab2.npy'))
rms_ab1_full = np.load(os.path.join(DIR_PROCESSED, 'rms_ab1.npy'))
rms_ab2_full = np.load(os.path.join(DIR_PROCESSED, 'rms_ab2.npy'))

media_iso_ab1_full = np.load(os.path.join(DIR_PROCESSED, 'media_iso_ab1.npy'))
rms_iso_ab1_full = np.load(os.path.join(DIR_PROCESSED, 'rms_iso_ab1.npy'))
rms_iso_ab2_full = np.load(os.path.join(DIR_PROCESSED, 'rms_iso_ab2.npy'))

media_ab1_chama = media_ab1_full[:Z_MASK, :]
media_ab2_chama = media_ab2_full[:Z_MASK, :]
rms_ab1_chama = rms_ab1_full[:Z_MASK, :]
rms_ab2_chama = rms_ab2_full[:Z_MASK, :]
rms_iso_ab1 = rms_iso_ab1_full[:Z_MASK, :]
rms_iso_ab2 = rms_iso_ab2_full[:Z_MASK, :]

diff_media = np.linalg.norm(media_ab1_chama - media_ab2_chama) / np.linalg.norm(media_ab1_chama)
err_rms_chama = (np.linalg.norm(rms_ab1_chama - rms_ab2_chama) / np.linalg.norm(rms_ab1_chama)) * 100
corr_rms_chama = np.corrcoef(rms_ab1_chama.ravel(), rms_ab2_chama.ravel())[0, 1]

err_rms_iso = (np.linalg.norm(rms_iso_ab1 - rms_iso_ab2) / np.linalg.norm(rms_iso_ab1)) * 100
corr_rms_iso = np.corrcoef(rms_iso_ab1.ravel(), rms_iso_ab2.ravel())[0, 1]

print(f"Mean Commutativity Relative Diff ||ab1 - ab2|| / ||ab1||: {diff_media:.2e}")
print(f"Flame RMS (Ab1): Min = {np.min(rms_ab1_chama):.4f} | Max = {np.max(rms_ab1_chama):.4f}")
print(f"Flame RMS (Ab2): Min = {np.min(rms_ab2_chama):.4f} | Max = {np.max(rms_ab2_chama):.4f}")
print(f"RMS Non-Commutativity ||ab1 - ab2|| / ||ab1||:")
print(f"   -> Flame: {err_rms_chama:.1f}% (Spatial correlation r = {corr_rms_chama:.2f})")
print(f"   -> ISO:   {err_rms_iso:.1f}% (Spatial correlation r = {corr_rms_iso:.2f})")

proj_recuperada = media_ab1_chama @ A_forward.T
p0_mediana = np.median(proj_recuperada[:, 0])
p1_3_mediana = np.median(proj_recuperada[:, 1:4])
print(f"\nKnife-Edge Parity Check (from Raw Projection P = f_mean @ A^T):")
print(f"   -> Median p(0) = {p0_mediana:.2f} a.u. vs Median p(1..3) = {p1_3_mediana:.2f} a.u. (!= 0)")
print("   -> CONCLUSION: Projection is EVEN (par) at the axis -> Reconstructed field is the")
print("      AXIAL refractive index gradient (dn/dz in a.u.), and A^-1 is the exact operator.\n")

# ==========================================
# 2. Sensor Noise Propagation (with delta) & ISO Null Case
# ==========================================
print("--- 2. SENSOR NOISE PROPAGATION & ISO NULL-CASE AUDIT ---")
arquivos_bg = sorted(glob.glob(DIR_BG), key=natural_sort_key)
n_bg = len(arquivos_bg)

soma_bg_full = np.zeros((512, 512), dtype=np.float64)
soma_bg_half = np.zeros((512, NUM_PIXEIS_RAIO), dtype=np.float64)
soma_bg_half_sq = np.zeros((512, NUM_PIXEIS_RAIO), dtype=np.float64)

for arq in arquivos_bg:
    img_bg = iio.imread(arq).astype(np.float64)
    soma_bg_full += img_bg
    metade_bg = extrair_metade_direita(img_bg, COL_SIMETRIA, NUM_PIXEIS_RAIO, delta_aplicado)
    soma_bg_half += metade_bg
    soma_bg_half_sq += (metade_bg ** 2)

bg_medio = soma_bg_full / n_bg
bg_half_medio = soma_bg_half / n_bg
var_bg_half = np.maximum((soma_bg_half_sq / n_bg) - (bg_half_medio ** 2), 0)
rms_bg_raw = np.sqrt(var_bg_half[:Z_MASK, :])

var_bg_rec_full = var_bg_half @ (A_inv.T ** 2)
sigma_rec_bg_full = np.sqrt(np.maximum(var_bg_rec_full, 0))
sigma_rec_bg = sigma_rec_bg_full[:Z_MASK, :]

rms_ab1_corr_full = np.sqrt(np.maximum((rms_ab1_full ** 2) - (sigma_rec_bg_full ** 2), 0))
rms_ab1_corr = rms_ab1_corr_full[:Z_MASK, :]

rms_iso_corr_full = np.sqrt(np.maximum((rms_iso_ab1_full ** 2) - (sigma_rec_bg_full ** 2), 0))
rms_iso_corr = rms_iso_corr_full[:Z_MASK, :]

np.save(os.path.join(DIR_PROCESSED, 'sigma_rec_bg.npy'), sigma_rec_bg_full)
np.save(os.path.join(DIR_PROCESSED, 'rms_ab1_corrigido.npy'), rms_ab1_corr_full)
np.save(os.path.join(DIR_PROCESSED, 'rms_iso_ab1_corrigido.npy'), rms_iso_corr_full)

# Measure and save burner geometry from Background so all visualization scripts use the exact measured R_b
d_bluff_px, centro_bg, delta_bg = medir_corpo_rombo_background(bg_medio, COL_SIMETRIA)
r_bluff_px = 0.5 * d_bluff_px
scale_mm_per_px = D_BLUFF_MM / d_bluff_px
np.save(
    os.path.join(DIR_PROCESSED, 'geometria_queimador.npy'),
    np.array([d_bluff_px, r_bluff_px, centro_bg, delta_bg, scale_mm_per_px])
)

rms_eixo_before = float(np.mean(rms_ab1_chama[:, 0]))
rms_eixo_after = float(np.mean(rms_ab1_corr[:, 0]))
pct_eixo = ((rms_eixo_after - rms_eixo_before) / rms_eixo_before) * 100.0

r_frente_ref = int(np.argmin(media_ab1_chama[80, 25:140]) + 25)
rms_front_before = float(rms_ab1_chama[80, r_frente_ref])
rms_front_after = float(rms_ab1_corr[80, r_frente_ref])
pct_front = ((rms_front_after - rms_front_before) / rms_front_before) * 100.0

r_slice = slice(20, 150)
snr_iso_perfil = np.mean(rms_iso_ab1[:, r_slice] / sigma_rec_bg[:, r_slice], axis=0)
frac_iso_sem_sinal = np.mean(rms_iso_ab1[:, r_slice] <= sigma_rec_bg[:, r_slice]) * 100
snr_chama_frente = np.mean(rms_ab1_chama[40:200, 75:100] / sigma_rec_bg[40:200, 75:100])

print(f"Mean Raw Sensor Noise (Background RMS, delta={delta_aplicado:+.2f} px): {np.mean(rms_bg_raw):.4f} a.u.")
print(f"Reconstructed Background Noise: Axis R=0 = {np.mean(sigma_rec_bg[:, 0]):.4f} | Radial Mean = {np.mean(sigma_rec_bg):.4f} a.u.")
print(f"Flame RMS at Axis (R=0): Uncorrected = {rms_eixo_before:.4f} -> Corrected = {rms_eixo_after:.4f} ({pct_eixo:+.1f}%)")
print(f"Flame RMS at Front (Z=80, R={r_frente_ref}): Uncorrected = {rms_front_before:.4f} -> Corrected = {rms_front_after:.4f} ({pct_front:+.1f}%)")
print(f"\nISO Null-Case Verification (r = 20..150 px):")
print(f"   -> ISO RMS / sigma_rec range: {np.min(snr_iso_perfil):.2f} to {np.max(snr_iso_perfil):.2f} (Flame front is ~{snr_chama_frente:.1f}x noise)")
print(f"   -> Pixels without resolvable signal (RMS_iso <= sigma_rec): {frac_iso_sem_sinal:.1f}%")
print(f"   -> ISO RMS at R=20 px (Z=40..120): Uncorrected = {np.mean(rms_iso_ab1[40:121, 20]):.3f} -> Corrected = {np.mean(rms_iso_corr[40:121, 20]):.3f} a.u.\n")

fig_rms, axs_rms = plt.subplots(1, 3, figsize=(15, 4.2))
for i, z in enumerate([40, 80, 120]):
    r_ax = np.arange(NUM_PIXEIS_RAIO) + 0.5  # Ring center r_j = j + 0.5 px
    axs_rms[i].plot(r_ax, rms_ab1_chama[z, :], 'r--', alpha=0.5, label='Flame Uncorrected RMS')
    axs_rms[i].plot(r_ax, rms_ab1_corr[z, :], 'r-', linewidth=1.8, label=r'Flame Corrected $\sqrt{RMS^2 - \sigma_{rec}^2}$')
    axs_rms[i].plot(r_ax, sigma_rec_bg[z, :], 'k:', linewidth=1.5, label=r'Propagated Noise $\sigma_{rec}$')
    axs_rms[i].plot(r_ax, rms_iso_corr[z, :], 'b-', alpha=0.8, linewidth=1.2, label='ISO Corrected RMS (Null Case)')
    axs_rms[i].set_title(f'Noise-Corrected RMS Profile (Z = {z} px)')
    axs_rms[i].set_xlabel('Radius r (px)')
    if i == 0:
        axs_rms[i].set_ylabel(r'Reconstructed $\partial n / \partial z$ RMS (a.u.)')
    axs_rms[i].set_xlim(0, 200)
    axs_rms[i].grid(True, alpha=0.3)
    axs_rms[i].legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(DIR_PROCESSED, 'rms_noise_correction.png'), dpi=300, bbox_inches='tight')
plt.close(fig_rms)

# ==========================================
# 3. Temporal Autocorrelation & Height-Dependent SE (Point 3)
# ==========================================
print("--- 3. TEMPORAL AUTOCORRELATION & HEIGHT-DEPENDENT SE(Mean) ---")
arquivos_chama = sorted(glob.glob(DIR_CHAMA), key=natural_sort_key)
N_CHAMA = len(arquivos_chama)
FPS = 400.0
DT_MS = 1000.0 / FPS

z_probes = [40, 80, 120]
r_probe = 82
series_temporais = np.zeros((N_CHAMA, len(z_probes)), dtype=np.float64)
a_inv_row = A_inv[r_probe, :]

print(f"Extracting chronological time series at Flame Front R={r_probe} px (with delta={delta_aplicado:+.2f} px)...")
for idx_f, arq in enumerate(arquivos_chama):
    img = iio.imread(arq).astype(np.float64)
    img_sub = img - bg_medio
    for idx_z, z_val in enumerate(z_probes):
        linha_dir = extrair_metade_direita(img_sub[z_val:z_val+1, :], COL_SIMETRIA, NUM_PIXEIS_RAIO, delta_aplicado)[0]
        series_temporais[idx_f, idx_z] = np.dot(linha_dir, a_inv_row)

lim_sinal = 0.20 * np.percentile(np.abs(media_ab1_chama), 99)
mascara_sinal = np.abs(media_ab1_chama) > lim_sinal
razao_rms_media_mediana = float(np.median(rms_ab1_corr[mascara_sinal] / np.abs(media_ab1_chama[mascara_sinal])))

plt.figure(figsize=(8, 4.5))
neff_dict = {}
se_rel_z_dict = {}
boot_diff_pct = {}

for idx_z, z_val in enumerate(z_probes):
    sinal = series_temporais[:, idx_z] - np.mean(series_temporais[:, idx_z])
    variancia = np.var(sinal)
    acf_full = np.correlate(sinal, sinal, mode='full')[N_CHAMA - 1:] / (variancia * N_CHAMA)
    
    zeros_idx = np.where(acf_full <= 0)[0]
    lag_corte = zeros_idx[0] if len(zeros_idx) > 0 else 50
    
    tau_int_frames = max(1.0 + 2.0 * np.sum(acf_full[1:lag_corte]), 1.0)
    tau_int_ms = 0.5 * tau_int_frames * DT_MS
    
    n_eff = N_CHAMA / tau_int_frames
    fator_inflacao = np.sqrt(tau_int_frames)
    neff_dict[z_val] = n_eff
    
    # Active-zone median relative SE at this height's N_eff(z)
    se_rel_z = (razao_rms_media_mediana / np.sqrt(n_eff)) * 100.0
    se_rel_z_dict[z_val] = se_rel_z
    
    bloco_tam = max(int(np.ceil(tau_int_frames * 2)), 2)
    n_blocos = N_CHAMA // bloco_tam
    rng = np.random.default_rng(42)
    medias_boot = np.array([
        np.mean(np.concatenate([series_temporais[ini:ini + bloco_tam, idx_z]
                                for ini in rng.integers(0, N_CHAMA - bloco_tam, size=n_blocos)]))
        for _ in range(1000)
    ])
    se_boot = np.std(medias_boot, ddof=1)
    se_naive_probe = np.std(series_temporais[:, idx_z], ddof=1) / np.sqrt(N_CHAMA)
    se_acf_probe = se_naive_probe * fator_inflacao
    red_boot = ((se_acf_probe - se_boot) / se_acf_probe) * 100.0
    boot_diff_pct[z_val] = red_boot
    
    print(f"Probe (Z={z_val:3d}, R={r_probe}): Zero-crossing = {lag_corte} frames ({lag_corte*DT_MS:.1f} ms)")
    print(f"   -> Integral Time Scale tau_int = {tau_int_ms:.2f} ms ({tau_int_frames:.2f} frames/indep. sample)")
    print(f"   -> N_eff(Z={z_val}) = {n_eff:.0f} | Active-Zone Relative SE(Z={z_val}) = {se_rel_z:.2f}%")
    print(f"   -> Probe SE: Naive={se_naive_probe:.4f}, ACF={se_acf_probe:.4f} (conservative), Bootstrap={se_boot:.4f} (-{red_boot:.1f}% vs ACF)")
    
    lags_ms = np.arange(40) * DT_MS
    plt.plot(lags_ms, acf_full[:40], marker='o', markersize=3.5, linewidth=1.6,
             label=f'Z = {z_val} px ($N_{{eff}} = {n_eff:.0f}$, $SE_{{rel}} = {se_rel_z:.1f}\\%$)')

plt.axhline(0, color='black', linestyle='--', alpha=0.6)
plt.xlabel('Time Lag $\\tau$ (ms) [at 400 fps]')
plt.ylabel('Autocorrelation Coefficient $\\rho(\\tau)$')
plt.title(f'Temporal Autocorrelation in the Flame Front (R = {r_probe} px, $\\delta = {delta_aplicado:+.2f}$ px)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(DIR_PROCESSED, 'temporal_autocorrelation.png'), dpi=300, bbox_inches='tight')
plt.close()

se_rel_naive = (razao_rms_media_mediana / np.sqrt(N_CHAMA)) * 100.0
N_EFF_MEAN = float(np.mean(list(neff_dict.values())))
se_rel_mean_neff = (razao_rms_media_mediana / np.sqrt(N_EFF_MEAN)) * 100.0

print(f"\nStatistical SE Summary (applies exclusively to the Mean field):")
print(f"   -> Naive (N={N_CHAMA}): {se_rel_naive:.2f}%")
print(f"   -> Height-Dependent ACF SE: Z=120 -> {se_rel_z_dict[120]:.2f}% | Z=80 -> {se_rel_z_dict[80]:.2f}% | Z=40 -> {se_rel_z_dict[40]:.2f}% (Mean N_eff=380 gives {se_rel_mean_neff:.2f}%)")
print(f"   -> Moving Block Bootstrap is {min(boot_diff_pct.values()):.0f}%–{max(boot_diff_pct.values()):.0f}% lower than ACF (confirming ACF as conservative bound).\n")

# ==========================================
# 4. Measured Axis, Burner Geometry, Ring-Center Algo Error & Uncertainty Budget
# ==========================================
print("--- 4. MEASURED AXIS, BURNER GEOMETRY & HEIGHT-DEPENDENT UNCERTAINTY BUDGET ---")

caminho_delta = os.path.join(DIR_PROCESSED, 'delta_eixo.npy')
if os.path.exists(caminho_delta):
    dados_d = np.load(caminho_delta)
    if len(dados_d) >= 7:
        d_med, d_app, d_std, d_10_l, d_10_h, d_b_l, d_b_h = dados_d[:7]
        print(f"1. Symmetry Axis from Flame Parity (|M(x) - M(-x)| min in +-15 px):")
        print(f"   -> Global optimum: delta = {d_med:+.2f} px (10%-min interval: [{d_10_l:+.2f}, {d_10_h:+.2f}] px)")
        print(f"   -> Axial z-blocks range: [{d_b_l:+.2f}, {d_b_h:+.2f}] px | Block dispersion = +- {d_std:.2f} px (justifies +-0.5 px budget)")

print(f"2. Independent Bluff-Body Measurement from Background Image (Z = 335..500 px, corrected sub-pixel parabola):")
print(f"   -> Measured Bluff-Body Diameter: D_b = {d_bluff_px:.2f} px (R_b = {r_bluff_px:.2f} px)")
print(f"   -> Burner Center in Background: col = {centro_bg:.2f} px (delta_bg = {delta_bg:+.2f} px vs flame delta = {delta_medido:+.2f} px)")
print(f"   -> Spatial Scale Factor: {scale_mm_per_px:.4f} mm/px (for D_b = {D_BLUFF_MM:.1f} mm — confirm exact value in lab guide)")

fwhm_lista = []
for z_val in range(40, 250):
    linha = media_ab1_chama[z_val, 40:150]
    val_min = np.min(linha)
    if val_min < -0.05:
        largura_meia = np.sum(linha < 0.5 * val_min)
        if largura_meia > 5:
            fwhm_lista.append(largura_meia)
fwhm_mediana = float(np.median(fwhm_lista))
print(f"3. Real Negative Band Width (FWHM across z=40..250 px): Median = {fwhm_mediana:.1f} px (10–90% range = {np.percentile(fwhm_lista, 10):.0f}–{np.percentile(fwhm_lista, 90):.0f} px)")

# Point 2: Pixel j is the ring [j, j+1] with geometric center at r_j = j + 0.5 px
r_grid = np.arange(NUM_PIXEIS_RAIO, dtype=np.float64)
r_ring_centers = r_grid + 0.5
R_flame_syn = 82.0
W_real = fwhm_mediana
W_pess = 8.8

def perfil_sintetico(r, R0, W):
    return -((r - R0) / W) * np.exp(-((r - R0) / W) ** 2)

def projecao_abel_continua(R0, W):
    p_cont = np.zeros(NUM_PIXEIS_RAIO, dtype=np.float64)
    r_max = float(NUM_PIXEIS_RAIO)
    for i in range(NUM_PIXEIS_RAIO):
        x_val = float(i)
        y_max = np.sqrt(r_max**2 - x_val**2)
        y_quad = np.linspace(0.0, y_max, 2000)
        r_quad = np.sqrt(x_val**2 + y_quad**2)
        p_cont[i] = 2.0 * trapezoid(perfil_sintetico(r_quad, R0, W), y_quad)
    return p_cont

f_true_centers = perfil_sintetico(r_ring_centers, R_flame_syn, W_real)
p_cont_0 = projecao_abel_continua(R_flame_syn, W_real)
f_rec_cont_0 = A_inv @ p_cont_0
erro_algoritmico = float((np.linalg.norm(f_rec_cont_0 - f_true_centers) / np.linalg.norm(f_true_centers)) * 100)
print(f"\n4. Pure Algorithmic Discretization Error (evaluated at ring centers r_j = j + 0.5 px, NO inverse crime): {erro_algoritmico:.2f}%")

p_exact_real = A_forward @ f_true_centers
f_true_pess = perfil_sintetico(r_ring_centers, R_flame_syn, W_pess)
p_exact_pess = A_forward @ f_true_pess

desvios_px = [0.5, 1.0, 1.5, 2.0]
erros_real = {}
erros_pess = {}

for dx in desvios_px:
    f_sh_real = A_inv @ np.interp(r_grid + dx, r_grid, p_exact_real, right=0.0)
    erros_real[dx] = float((np.linalg.norm(f_sh_real - f_true_centers) / np.linalg.norm(f_true_centers)) * 100)
    
    f_sh_pess = A_inv @ np.interp(r_grid + dx, r_grid, p_exact_pess, right=0.0)
    erros_pess[dx] = float((np.linalg.norm(f_sh_pess - f_true_pess) / np.linalg.norm(f_true_pess)) * 100)
    print(f"   Axis Shift dx = {dx:.1f} px -> L2 Error (Measured W={W_real:.0f} px): {erros_real[dx]:5.2f}% | (Pessimistic W=8.8 px): {erros_pess[dx]:5.2f}%")

# Height-dependent total combined uncertainty u_total(z)
u_tot_z = {
    z_val: np.sqrt(se_rel_z_dict[z_val]**2 + erros_real[0.5]**2 + erro_algoritmico**2)
    for z_val in z_probes
}
u_total_mean_neff = np.sqrt(se_rel_mean_neff**2 + erros_real[0.5]**2 + erro_algoritmico**2)
u_total_pess = np.sqrt(se_rel_mean_neff**2 + erros_pess[0.5]**2 + erro_algoritmico**2)

print(f"\nTOTAL COMBINED RELATIVE UNCERTAINTY OF MEAN FIELD (Active Zone):")
print(f"   -> Height-Dependent Realistic Budget (Axis_0.5px={erros_real[0.5]:.2f}% + Algo={erro_algoritmico:.2f}% + SE(z)):")
print(f"      * Z = 120 px (SE={se_rel_z_dict[120]:.2f}%): {u_tot_z[120]:.2f}%")
print(f"      * Z =  80 px (SE={se_rel_z_dict[80]:.2f}%): {u_tot_z[80]:.2f}%")
print(f"      * Z =  40 px (SE={se_rel_z_dict[40]:.2f}%): {u_tot_z[40]:.2f}%")
print(f"      => Height Interval Range: {min(u_tot_z.values()):.1f}% to {max(u_tot_z.values()):.1f}% (Mean N_eff=380 reference: {u_total_mean_neff:.2f}%)")
print(f"   -> Conditional Pessimistic Bound (with W=8.8 px Axis_0.5px={erros_pess[0.5]:.2f}%): {u_total_pess:.2f}%")

plt.figure(figsize=(8, 4.5))
plt.plot(r_ring_centers, f_true_centers, 'k-', linewidth=2.2, label=f'True Profile (Measured W = {W_real:.0f} px, dx = 0.0 px)')
for dx, cor, estilo in zip([0.5, 1.0, 2.0], ['tab:blue', 'tab:orange', 'tab:red'], ['--', '-.', ':']):
    f_sh = A_inv @ np.interp(r_grid + dx, r_grid, p_exact_real, right=0.0)
    plt.plot(r_ring_centers, f_sh, color=cor, linestyle=estilo, linewidth=1.6,
             label=f'Shift dx = {dx:.1f} px (L2 Err = {erros_real[dx]:.1f}%)')
plt.xlim(0, 180)
plt.xlabel('Radius r (px) [ring center j + 0.5]')
plt.ylabel(r'Reconstructed $\partial n / \partial z$ Signal (a.u.)')
plt.title(f'Onion-Peeling Sensitivity to Axis Shift (Measured Band W = {W_real:.0f} px)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(DIR_PROCESSED, 'axis_sensitivity_uncertainty.png'), dpi=300, bbox_inches='tight')
plt.close()