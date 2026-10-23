import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Configurar caminhos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')

# 2. Carregar as matrizes do caso ISO gravadas pelo approach2.py
media_ab1 = np.load(os.path.join(DIR_PROCESSED, 'media_iso_ab1.npy'))
rms_ab1 = np.load(os.path.join(DIR_PROCESSED, 'rms_iso_ab1.npy'))
media_ab2 = np.load(os.path.join(DIR_PROCESSED, 'media_iso_ab2.npy'))
rms_ab2 = np.load(os.path.join(DIR_PROCESSED, 'rms_iso_ab2.npy'))

# 3. Preparar a figura para comparação (2x2)
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

def plot_campo(ax, matriz, titulo, cmap='viridis'):
    # Para o caso ISO (sem combustão), o 'viridis' (tons de verde/azul/amarelo) 
    # ajuda a distinguir visualmente da Chama ('magma' - tons de fogo)
    im = ax.imshow(matriz, cmap=cmap) 
    ax.set_title(titulo, fontsize=14)
    ax.set_xlabel('Raio (px)')
    ax.set_ylabel('Altura Z (px)')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# 4. Desenhar os 4 gráficos
plot_campo(axs[0, 0], media_ab1, "Média ISO (Abordagem 1: Tomo -> Stats)")
plot_campo(axs[0, 1], media_ab2, "Média ISO (Abordagem 2: Stats -> Tomo)")
plot_campo(axs[1, 0], rms_ab1, "RMS ISO (Abordagem 1)", cmap='cividis')
plot_campo(axs[1, 1], rms_ab2, "RMS ISO (Abordagem 2)", cmap='cividis')

plt.tight_layout()

# 5. Mostrar no ecrã e guardar a imagem
caminho_imagem = os.path.join(DIR_PROCESSED, 'comparacao_iso.png')
plt.savefig(caminho_imagem, dpi=300, bbox_inches='tight')
print(f"Imagem guardada em: {caminho_imagem}")

plt.show()
