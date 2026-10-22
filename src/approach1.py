import numpy as np
import glob
import imageio.v3 as iio
import os
from utils import calcular_matriz_ART_inversa

# ==========================================
# 1. Configurações e Caminhos Absolutos
# ==========================================
# Deteta automaticamente a pasta onde está este script (src/) e sobe um nível para o projeto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DIR_BG = os.path.join(BASE_DIR, 'data', 'raw', 'Background', 'tiff', '*.tif')
DIR_CHAMA = os.path.join(BASE_DIR, 'data', 'raw', 'Chama', '*.tif')
DIR_PROCESSED = os.path.join(BASE_DIR, 'data', 'processed')

# Garante que a pasta de destino existe
os.makedirs(DIR_PROCESSED, exist_ok=True)

def calcular_background_medio(padrao_arquivos):
    arquivos = glob.glob(padrao_arquivos)
    n_frames = len(arquivos)
    
    if n_frames == 0:
        raise ValueError(f"ERRO: Nenhuma imagem encontrada no caminho {padrao_arquivos}. Confirma se extraíste os ficheiros corretamente.")
        
    print(f"A calcular fundo médio ({n_frames} imagens)...")
    soma_bg = np.zeros((512, 512), dtype=np.float64)
    
    for arquivo in arquivos:
        img = iio.imread(arquivo).astype(np.float64)
        soma_bg += img
        
    print("Fundo médio calculado com sucesso!")
    return soma_bg / n_frames

def aplicar_tomografia(imagem_metade, A_inv):
    # O produto matricial aplica a reconstrução a todas as linhas horizontais em simultâneo
    return imagem_metade @ A_inv.T

# ==========================================
# 2. Execução da Abordagem 1 (Tomo -> Média/RMS)
# ==========================================
bg_medio = calcular_background_medio(DIR_BG)
arquivos_chama = glob.glob(DIR_CHAMA)
n_chama = len(arquivos_chama)

# O eixo de simetria precisa de ser identificado (exemplo: coluna 256)
col_simetria = 256
num_pixeis_raio = 512 - col_simetria
A_inv_precalculada = calcular_matriz_ART_inversa(num_pixeis_raio)

# Acumuladores para Média e RMS (apenas da metade reconstruída)
shape_metade = (512, 512 - col_simetria)
soma_tomo = np.zeros(shape_metade, dtype=np.float64)
soma_tomo_sq = np.zeros(shape_metade, dtype=np.float64)

for arquivo in arquivos_chama:
    # 1. Ler imagem e converter para float
    img = iio.imread(arquivo).astype(np.float64)
    
    # 2. Subtrair background (evitar valores negativos espúrios de ruído)
    img_sub = img - bg_medio
    img_sub[img_sub < 0] = 0 
    
    # 3. Cortar metade direita a partir do eixo de simetria
    metade_dir = img_sub[:, col_simetria:]
    
    # 4. ABORDAGEM 1: Tomografia IMEDIATA em cada instante
    tomo_inst = aplicar_tomografia(metade_dir, A_inv_precalculada)
    
    # 5. Acumular para a estatística
    soma_tomo += tomo_inst
    soma_tomo_sq += (tomo_inst ** 2)

# Cálculos Finais - Abordagem 1
media_ab1 = soma_tomo / n_chama
rms_ab1 = np.sqrt((soma_tomo_sq / n_chama) - (media_ab1 ** 2))

# ==========================================
# 3. Execução da Abordagem 2 (Média/RMS -> Tomo)
# ==========================================
# Inicializar acumuladores diretamente para os dados em bruto subtraídos
soma_raw = np.zeros(shape_metade, dtype=np.float64)
soma_raw_sq = np.zeros(shape_metade, dtype=np.float64)

for arquivo in arquivos_chama:
    img = iio.imread(arquivo).astype(np.float64)
    img_sub = img - bg_medio
    img_sub[img_sub < 0] = 0
    metade_dir = img_sub[:, col_simetria:]
    
    # Acumular ESTATÍSTICA ANTES da tomografia
    soma_raw += metade_dir
    soma_raw_sq += (metade_dir ** 2)

# 1. Calcular Média e RMS das projeções integradas (Bruto)
media_raw = soma_raw / n_chama
rms_raw = np.sqrt((soma_raw_sq / n_chama) - (media_raw ** 2))

# 2. ABORDAGEM 2: Aplicar Tomografia ÚNICA no final
media_ab2 = aplicar_tomografia(media_raw, A_inv_precalculada)
rms_ab2 = aplicar_tomografia(rms_raw, A_inv_precalculada)

# Salvar resultados na pasta processed
caminho_media_ab1 = os.path.join(DIR_PROCESSED, 'media_ab1.npy')
caminho_rms_ab1 = os.path.join(DIR_PROCESSED, 'rms_ab1.npy')
# Caminhos para Abordagem 2
caminho_media_ab2 = os.path.join(DIR_PROCESSED, 'media_ab2.npy')
caminho_rms_ab2 = os.path.join(DIR_PROCESSED, 'rms_ab2.npy')

np.save(caminho_media_ab1, media_ab1)
np.save(caminho_rms_ab1, rms_ab1)
np.save(caminho_media_ab2, media_ab2)
np.save(caminho_rms_ab2, rms_ab2)

print(f"Concluído! Ficheiros gravados na pasta: {DIR_PROCESSED}")