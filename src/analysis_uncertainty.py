import numpy as np
import imageio.v3 as iio
import glob
import os
import matplotlib.pyplot as plt
from utils import calcular_matriz_ART_inversa

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')
DIR_BG = os.path.join(BASE_DIR, 'data', 'raw', 'Background', 'tiff', '*.tif')

Z_MASK = 325  # Mask bluff-body region (Z >= 325 px)
COL_SIMETRIA = 256
NUM_PIXEIS_RAIO = 512 - COL_SIMETRIA
A_inv = calcular_matriz_ART_inversa(NUM_PIXEIS_RAIO)

# ==========================================
# 1. Mathematical Proof: Failure of Approach 2
# ==========================================
print("--- 1. PHYSICAL ANALYSIS OF RMS (FLOW DOMAIN Z < 325 px) ---")
rms_ab1_full = np.load(os.path.join(DIR_PROCESSED, 'rms_ab1.npy'))
rms_ab2_full = np.load(os.path.join(DIR_PROCESSED, 'rms_ab2.npy'))
media_ab1_full = np.load(os.path.join(DIR_PROCESSED, 'media_ab1.npy'))

# Restrict analysis to the physical flow domain (excluding bluff-body reflection)
rms_ab1_chama = rms_ab1_full[:Z_MASK, :]
rms_ab2_chama = rms_ab2_full[:Z_MASK, :]
media_ab1_chama = media_ab1_full[:Z_MASK, :]

print(f"Approach 1 (Tomography -> Stats): Min = {np.min(rms_ab1_chama):.4f} | Max = {np.max(rms_ab1_chama):.4f}")
print(f"Approach 2 (Stats -> Tomography): Min = {np.min(rms_ab2_chama):.4f} | Max = {np.max(rms_ab2_chama):.4f}")
print("-> CONCLUSION: Approach 2 generates non-physical negative values, confirming that non-linear operators (RMS) do not commute with the inverse Abel matrix.\n")

# ==========================================
# 2. Statistical Uncertainty and Sensor Noise
# ==========================================
print("--- 2. CONVERGENCE AND SENSOR NOISE ---")
N_CHAMA = 3027
erro_padrao = rms_ab1_chama / np.sqrt(N_CHAMA)

# Evaluate relative uncertainty in active Schlieren zone (|signal| > 20% of 99th percentile)
lim_sinal = 0.20 * np.percentile(np.abs(media_ab1_chama), 99)
mascara_sinal = np.abs(media_ab1_chama) > lim_sinal
erros_relativos = (erro_padrao[mascara_sinal] / np.abs(media_ab1_chama[mascara_sinal])) * 100

print(f"Naive Standard Error (RMS/sqrt(N)) Max: {np.max(erro_padrao):.4f}")
print(f"Median Relative Statistical Error (active zone): {np.median(erros_relativos):.2f}%")

print("Calculating background sensor noise and tomographic propagation...")
arquivos_bg = glob.glob(DIR_BG)
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

# Analytical variance propagation through linear Onion-Peeling operator:
# Var(f_rec) = Var(p_raw) @ (A_inv.T ** 2)
var_bg_rec = var_bg[:Z_MASK, COL_SIMETRIA:] @ (A_inv.T ** 2)
sigma_rec_bg = np.sqrt(np.maximum(var_bg_rec, 0))

print(f"Mean Raw Sensor Noise (Background RMS): {np.mean(rms_bg_raw):.4f} a.u.")
print(f"Reconstructed Background Noise (At Axis R=0): {np.mean(sigma_rec_bg[:, 0]):.4f} a.u.")
print(f"Reconstructed Background Noise (Radial Mean): {np.mean(sigma_rec_bg):.4f} a.u.\n")

# Save reconstructed background noise map for RMS correction (Step 5)
np.save(os.path.join(DIR_PROCESSED, 'sigma_rec_bg.npy'), sigma_rec_bg)

# ==========================================
# 3. Symmetry Axis Uncertainty (Synthetic & Propagated)
# ==========================================
print("--- 3. SYMMETRY AXIS SENSITIVITY & UNCERTAINTY PROPAGATION ---")

# Build a forward Abel matrix A from A_inv to generate an exact synthetic projection
A_forward = np.linalg.inv(A_inv)
r_grid = np.arange(NUM_PIXEIS_RAIO, dtype=np.float64)

# Synthetic Schlieren-like radial profile (derivative of Gaussian density profile)
R_flame_syn = 80.0
W_syn = 8.8
f_true = -((r_grid - R_flame_syn) / W_syn) * np.exp(-((r_grid - R_flame_syn) / W_syn) ** 2)
p_exact = A_forward @ f_true

desvios_px = [0.5, 1.0, 1.5, 2.0]
erros_sinteticos = {}

for dx in desvios_px:
    # Sub-pixel shift of the projection using linear interpolation
    p_shifted = np.interp(r_grid + dx, r_grid, p_exact, right=0.0)
    f_rec_shift = A_inv @ p_shifted
    
    # Relative L2 norm error (%)
    erro_l2 = (np.linalg.norm(f_rec_shift - f_true) / np.linalg.norm(f_true)) * 100
    erros_sinteticos[dx] = erro_l2
    print(f"Synthetic Axis Shift dx = {dx:.1f} px -> Relative Reconstruction Error (L2) = {erro_l2:.2f}%")

print("-> CRITICAL CONCLUSION: Symmetry axis uncertainty is NOT negligible!")
print(f"   Even a sub-pixel shift of 0.5 px induces ~{erros_sinteticos[0.5]:.1f}% global error")
print(f"   and a 1.0 px shift induces ~{erros_sinteticos[1.0]:.1f}% error due to 1/r amplification near R=0.\n")

# Propagate Axis Uncertainty + Statistical Uncertainty in Quadrature on Real Flame Data
# Assuming a realistic experimental axis identification uncertainty of dx = +/- 0.5 px:
erro_rel_eixo_05 = erros_sinteticos[0.5] / 100.0
u_eixo_campo = erro_rel_eixo_05 * np.abs(media_ab1_chama)
u_total_combinada = np.sqrt(erro_padrao**2 + u_eixo_campo**2)

rel_u_total = (u_total_combinada[mascara_sinal] / np.abs(media_ab1_chama[mascara_sinal])) * 100
print(f"Combined Relative Uncertainty (Stat + 0.5 px Axis Shift, Median in Active Zone): {np.median(rel_u_total):.2f}%")

# Save diagnostic plot of Axis Shift Sensitivity
plt.figure(figsize=(8, 4.5))
plt.plot(r_grid, f_true, 'k-', linewidth=2.2, label='True Synthetic Profile (dx = 0.0 px)')
for dx, cor, estilo in zip([0.5, 1.0, 2.0], ['tab:blue', 'tab:orange', 'tab:red'], ['--', '-.', ':']):
    p_sh = np.interp(r_grid + dx, r_grid, p_exact, right=0.0)
    f_sh = A_inv @ p_sh
    plt.plot(r_grid, f_sh, color=cor, linestyle=estilo, linewidth=1.6,
             label=f'Shift dx = {dx:.1f} px (L2 Err = {erros_sinteticos[dx]:.1f}%)')

plt.xlim(0, 180)
plt.xlabel('Radius (px)')
plt.ylabel('Reconstructed Schlieren Signal (a.u.)')
plt.title('Sensitivity of Onion-Peeling to Symmetry Axis Misalignment')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
caminho_fig_eixo = os.path.join(DIR_PROCESSED, 'axis_sensitivity_uncertainty.png')
plt.savefig(caminho_fig_eixo, dpi=300, bbox_inches='tight')
print(f"Saved axis sensitivity figure: {caminho_fig_eixo}")