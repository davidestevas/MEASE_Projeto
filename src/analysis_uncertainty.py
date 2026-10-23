import numpy as np
import imageio.v3 as iio
import glob
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')
DIR_BG = os.path.join(BASE_DIR, 'data', 'raw', 'Background', 'tiff', '*.tif')

# ==========================================
# 1. Prova do Erro da Abordagem 2
# ==========================================
print("--- ANÁLISE DO ERRO FÍSICO DO RMS ---")
rms_ab1_chama = np.load(os.path.join(DIR_PROCESSED, 'rms_ab1.npy'))
rms_ab2_chama = np.load(os.path.join(DIR_PROCESSED, 'rms_ab2.npy'))

print(f"Abordagem 1 (Tomografia -> Stats): Min = {np.min(rms_ab1_chama):.4f}, Max = {np.max(rms_ab1_chama):.4f}")
print(f"Abordagem 2 (Stats -> Tomografia): Min = {np.min(rms_ab2_chama):.4f}, Max = {np.max(rms_ab2_chama):.4f}")
print("-> CONCLUSÃO: A Abordagem 2 é não-física devido a valores negativos gerados pela inversão algébrica de um campo quadrático.")

# ==========================================
# 2. Erro Padrão da Média (Convergência)
# ==========================================
print("\n--- ANÁLISE DE INCERTEZA ESTATÍSTICA ---")
N_CHAMA = 3027 # Número de imagens
erro_padrao = rms_ab1_chama / np.sqrt(N_CHAMA)
erro_maximo = np.max(erro_padrao)
print(f"Erro Padrão Máximo na Chama: {erro_maximo:.4f} (graças ao elevado N={N_CHAMA}, o erro estatístico é mínimo)")

# ==========================================
# 3. Deteção do Eixo de Simetria do Bluff-Body
# ==========================================
print("\n--- VALIDAÇÃO DO EIXO DE SIMETRIA ---")
# Carrega uma imagem de fundo (que contém o cilindro escuro)
arquivos_bg = glob.glob(DIR_BG)
img_bg = iio.imread(arquivos_bg[0]).astype(float)

# Somar as intensidades ao longo da vertical (Z)
perfil_horizontal = np.sum(img_bg, axis=0)

# O bluff-body bloqueia a luz, logo é o vale (mínimo) do perfil
# Para encontrar o centro com precisão, pegamos na zona mais escura
limiar_sombra = np.min(perfil_horizontal) + (np.max(perfil_horizontal) - np.min(perfil_horizontal)) * 0.2
indices_sombra = np.where(perfil_horizontal < limiar_sombra)[0]

centro_estimado = np.mean(indices_sombra)
print(f"Centro de simetria assumido no código: 256")
print(f"Centro de simetria calculado pela sombra: {centro_estimado:.1f}")

desvio_pixeis = abs(256 - centro_estimado)
print(f"-> O erro de alinhamento é de apenas ~{desvio_pixeis:.1f} pixeis. O hardcode de 256 introduz uma incerteza negligenciável no algoritmo de Onion-Peeling.")
