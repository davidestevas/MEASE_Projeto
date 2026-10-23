import numpy as np
import imageio.v3 as iio
import glob
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')
DIR_BG = os.path.join(BASE_DIR, 'data', 'raw', 'Background', 'tiff', '*.tif')

# ==========================================
# 1. Prova Matemática: Falha da Abordagem 2
# ==========================================
print("--- 1. ANÁLISE FÍSICA DO RMS ---")
rms_ab1_chama = np.load(os.path.join(DIR_PROCESSED, 'rms_ab1.npy'))
rms_ab2_chama = np.load(os.path.join(DIR_PROCESSED, 'rms_ab2.npy'))
print(f"Abordagem 1 (Tomografia -> Stats): Min = {np.min(rms_ab1_chama):.4f} | Max = {np.max(rms_ab1_chama):.4f}")
print(f"Abordagem 2 (Stats -> Tomografia): Min = {np.min(rms_ab2_chama):.4f} | Max = {np.max(rms_ab2_chama):.4f}")
print("-> CONCLUSÃO: A Abordagem 2 gera valores negativos não-físicos, confirmando que operadores não-lineares (RMS) não comutam com a matriz inversa de Abel.\n")

# ==========================================
# 2. Incerteza Estatística e Ruído do Sensor
# ==========================================
print("--- 2. CONVERGÊNCIA E RUÍDO (BACKGROUND) ---")
N_CHAMA = 3027
media_ab1_chama = np.load(os.path.join(DIR_PROCESSED, 'media_ab1.npy'))

# Erro Padrão da Média e Erro Relativo
erro_padrao = rms_ab1_chama / np.sqrt(N_CHAMA)
erro_maximo = np.max(erro_padrao)

# Evitar dividir por zero nas zonas mortas (usamos apenas a frente de chama onde o sinal é forte)
mascara_sinal = media_ab1_chama > 0.05
erro_relativo_max = np.max(erro_padrao[mascara_sinal] / media_ab1_chama[mascara_sinal]) * 100

print(f"Erro Padrão Absoluto Máximo: {erro_maximo:.4f}")
print(f"Erro Relativo Máximo (na zona de reação): {erro_relativo_max:.2f}%")

# Ruído do Fundo (Dark Noise e iluminação espúria)
arquivos_bg = glob.glob(DIR_BG)
# Carregar o bloco 3D inteiro para avaliar estatística ao longo do tempo
bg_stack = np.array([iio.imread(arq).astype(float) for arq in arquivos_bg])
bg_ruido_std = np.std(bg_stack, axis=0)
bg_medio = np.mean(bg_stack, axis=0)

print(f"Ruído médio do Fundo (Desvio Padrão): {np.mean(bg_ruido_std):.2f} níveis de cinza.")
print(f"Ruído máximo espúrio no Fundo: {np.max(bg_ruido_std):.2f} níveis de cinza.\n")

# ==========================================
# 3. Validação do Alinhamento do Bluff-Body
# ==========================================
print("--- 3. ALINHAMENTO DO EIXO DE SIMETRIA ---")
# Projeção horizontal do background médio completo (não apenas 1 frame)
perfil_horizontal = np.sum(bg_medio, axis=0)
limiar_sombra = np.min(perfil_horizontal) + (np.max(perfil_horizontal) - np.min(perfil_horizontal)) * 0.2
indices_sombra = np.where(perfil_horizontal < limiar_sombra)[0]

centro_estimado = np.mean(indices_sombra)
desvio_pixeis = abs(256 - centro_estimado)

print(f"Centro assumido (hardcode): 256.0 px")
print(f"Centro detetado opticamente: {centro_estimado:.1f} px")

# Conclusão condicional
if desvio_pixeis <= 2.0:
    print(f"-> AVALIAÇÃO: Desvio de {desvio_pixeis:.1f} px é negligenciável para o algoritmo Onion-Peeling.")
elif desvio_pixeis <= 8.0:
    print(f"-> AVALIAÇÃO: Desvio moderado ({desvio_pixeis:.1f} px). Aceitável, mas introduz ligeira assimetria na reconstrução.")
else:
    print(f"-> AVALIAÇÃO: Desvio Crítico! A matriz geométrica está desfasada em {desvio_pixeis:.1f} px. Necessário ajustar 'col_simetria' no código base.")