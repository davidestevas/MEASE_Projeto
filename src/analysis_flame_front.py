import numpy as np
import matplotlib.pyplot as plt
import os

# Caminhos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')

# Carregar dados da Abordagem 1 (A correta para o RMS)
media_chama = np.load(os.path.join(DIR_PROCESSED, 'media_ab1.npy'))
rms_chama = np.load(os.path.join(DIR_PROCESSED, 'rms_ab1.npy'))
media_iso = np.load(os.path.join(DIR_PROCESSED, 'media_iso_ab1.npy'))
rms_iso = np.load(os.path.join(DIR_PROCESSED, 'rms_iso_ab1.npy'))

# Escolher alturas Z para extrair os perfis (em pixéis)
# Assumindo que Z=0 é o topo da imagem e o bluff-body está na base
alturas_z = [100, 250, 400] 

fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for i, z in enumerate(alturas_z):
    raio = np.arange(rms_chama.shape[1])
    
    # Perfis de RMS
    perfil_rms_chama = rms_chama[z, :]
    perfil_rms_iso = rms_iso[z, :]
    
    # Encontrar a posição radial (frente de chama) como o pico do RMS
    pico_chama_r = np.argmax(perfil_rms_chama)
    pico_iso_r = np.argmax(perfil_rms_iso)
    
    axs[i].plot(raio, perfil_rms_chama, 'r-', label='Chama (RMS)', linewidth=2)
    axs[i].plot(raio, perfil_rms_iso, 'b--', label='ISO (RMS)', linewidth=2)
    
    # Marcar os picos
    axs[i].axvline(pico_chama_r, color='red', alpha=0.3, linestyle=':')
    axs[i].axvline(pico_iso_r, color='blue', alpha=0.3, linestyle=':')
    
    axs[i].set_title(f'Perfil Radial Z = {z} px')
    axs[i].set_xlabel('Raio (px)')
    if i == 0:
        axs[i].set_ylabel('Intensidade de Flutuação (RMS)')
    axs[i].legend()
    
    print(f"Altura Z={z}: Frente de Chama em R={pico_chama_r}px | Shear Layer ISO em R={pico_iso_r}px")

plt.tight_layout()
plt.savefig(os.path.join(DIR_PROCESSED, 'perfis_radiais_chama_vs_iso.png'), dpi=300)
print("Gráfico de localização da chama guardado!")
plt.show()