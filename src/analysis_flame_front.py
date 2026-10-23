import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks

# 1. Configurar caminhos e carregar dados corretos (Abordagem 1)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')

rms_chama = np.load(os.path.join(DIR_PROCESSED, 'rms_ab1.npy'))
rms_iso = np.load(os.path.join(DIR_PROCESSED, 'rms_iso_ab1.npy'))

# 2. Parâmetros físicos corrigidos
alturas_z = [40, 80, 120]  # Zonas de forte reação na metade superior da imagem
RAIO_MIN = 15              # Cortar o artefato do Onion-Peeling no eixo (R=0)

fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

print("--- LOCALIZAÇÃO DA FRENTE DE CHAMA E SHEAR LAYER ---")
for i, z in enumerate(alturas_z):
    raio = np.arange(rms_chama.shape[1])
    
    perfil_rms_chama = rms_chama[z, :]
    perfil_rms_iso = rms_iso[z, :]
    
    # Detetar pico da Chama
    picos_c, props_c = find_peaks(perfil_rms_chama[RAIO_MIN:], prominence=0.02)
    if len(picos_c) > 0:
        p_c = picos_c[np.argmax(props_c['prominences'])] + RAIO_MIN
        print(f"Z={z} px | Chama: R={p_c} px (RMS={rms_chama[z, p_c]:.3f})")
    else:
        p_c = None
        
    # Detetar pico ISO
    picos_i, props_i = find_peaks(perfil_rms_iso[RAIO_MIN:], prominence=0.02)
    if len(picos_i) > 0:
        p_i = picos_i[np.argmax(props_i['prominences'])] + RAIO_MIN
        print(f"Z={z} px | ISO: R={p_i} px (RMS={rms_iso[z, p_i]:.3f})")
    else:
        p_i = None

    # Desenhar os perfis
    axs[i].plot(raio, perfil_rms_chama, 'r-', label='Chama (RMS)', linewidth=2)
    axs[i].plot(raio, perfil_rms_iso, 'b--', label='ISO (RMS)', linewidth=2)
    
    if p_c is not None:
        axs[i].axvline(p_c, color='red', alpha=0.5, linestyle=':', label=f'Chama R={p_c}')
    if p_i is not None:
        axs[i].axvline(p_i, color='blue', alpha=0.5, linestyle=':', label=f'ISO R={p_i}')
    
    axs[i].set_title(f'Perfil Radial Z = {z} px')
    axs[i].set_xlabel('Raio (px)')
    if i == 0:
        axs[i].set_ylabel('Intensidade de Flutuação (RMS)')
    
    axs[i].set_xlim(0, rms_chama.shape[1])
    axs[i].legend()

plt.tight_layout()
caminho_figura = os.path.join(DIR_PROCESSED, 'perfis_radiais_corrigidos.png')
plt.savefig(caminho_figura, dpi=300)
print(f"Gráfico guardado em: {caminho_figura}")
plt.show()