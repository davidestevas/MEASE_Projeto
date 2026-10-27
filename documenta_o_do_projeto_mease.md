# Projeto MEASE — Processamento de Imagem, Análise de Chama e Escoamento Isotérmico

Este repositório contém o código-fonte e os resultados processados referentes ao projeto da unidade curricular de **MEASE**. O trabalho foca-se no processamento estatístico de imagens experimentais para a caracterização de uma frente de chama (condições reativas) e de um jato/escoamento isotérmico (condições não-reativas), bem como na análise rigorosa de incertezas, correção de ruído de fundo e avaliação do efeito de *clipping*.

---

## Objetivos do Trabalho

O projeto está dividido em três grandes pilares de análise experimental:

1. **Processamento Base (Chama vs. Isotérmico):**
   - Cálculo de campos médios temporais (`media`) e de flutuação (`rms` - *Root Mean Square*) da intensidade luminosa / sinal adquirido para diferentes aberturas ou configurações óticas (`ab1` e `ab2`).
2. **Análise de Incertezas e Viés Estatístico:**
   - **Efeito de *Clipping* (Truncatura):** Avaliação do impacto de truncar valores negativos de intensidade (resultantes da subtração do fundo) a zero, comparando o processamento com e sem truncatura.
   - **Correção de Ruído de Fundo no RMS:** Isolamento da flutuação real do fenómeno físico através da subtração quadrática da variância do ruído de fundo ($\sigma_{\text{bg}}^2$).
   - **Autocorrelação Temporal e Incerteza:** Estudo da independência estatística entre *frames* consecutivos para determinar o número efetivo de amostras independentes ($N_{\text{eff}}$) e quantificar a incerteza associada aos campos médios e ao RMS.
3. **Caracterização Física:**
   - Deteção e análise da evolução espacial da **frente de chama** e estudo do **decaimento axial/radial** no escoamento isotérmico.

---

## Estrutura do Repositório

```text
MEASE_Projeto/
├── data/
│   ├── raw/                             # Dados brutos originais (imagens/matrizes experimentais - ignorados no Git)
│   └── processed/                       # Matrizes processadas (.npy) e gráficos gerados (.png)
│       ├── media_ab1.npy                # Campo médio da chama (Configuração/Abertura 1)
│       ├── media_ab2.npy                # Campo médio da chama (Configuração/Abertura 2)
│       ├── rms_ab1.npy                  # Campo RMS da chama sem correção de ruído (Abertura 1)
│       ├── rms_ab2.npy                  # Campo RMS da chama sem correção de ruído (Abertura 2)
│       ├── rms_ab1_corrigido.npy        # Campo RMS da chama corrigido do ruído de fundo (Abertura 1)
│       ├── media_iso_ab1.npy            # Campo médio do escoamento isotérmico (Abertura 1)
│       ├── media_iso_ab2.npy            # Campo médio do escoamento isotérmico (Abertura 2)
│       ├── rms_iso_ab1.npy              # Campo RMS isotérmico sem correção (Abertura 1)
│       ├── rms_iso_ab2.npy              # Campo RMS isotérmico sem correção (Abertura 2)
│       ├── rms_iso_ab1_corrigido.npy    # Campo RMS isotérmico corrigido do ruído (Abertura 1)
│       ├── sigma_rec_bg.npy             # Matriz com o desvio padrão do ruído de fundo reconstruído
│       ├── comparacao_chama.png         # Visualização comparativa dos campos da chama (ab1 vs ab2)
│       ├── comparacao_iso.png           # Visualização comparativa dos campos isotérmicos (ab1 vs ab2)
│       ├── clipping_comparison.png      # Gráfico do impacto do viés de truncatura (Clipping Bias)
│       ├── rms_noise_correction.png     # Comparação do perfil/campo RMS antes e após correção de ruído
│       ├── temporal_autocorrelation.png # Curvas de autocorrelação temporal das amostras
│       └── flame_front_and_iso_decay.png# Análise da posição da frente de chama e decaimento isotérmico
│
├── src/                                 # Scripts de processamento, análise e visualização
│   ├── utils.py                         # Funções auxiliares de leitura de ficheiros, pré-processamento e matemática
│   ├── process_flame.py                 # Cálculo das médias e RMS para os ensaios de chama
│   ├── process_iso.py                   # Cálculo das médias e RMS para os ensaios isotérmicos
│   ├── visualize.py                     # Geração dos mapas de cor e perfis para os ensaios de chama
│   ├── visualize_iso.py                 # Geração dos mapas de cor e perfis para os ensaios isotérmicos
│   ├── analysis_clipping_bias.py        # Estudo do erro induzido pelo clipping de valores negativos
│   ├── analysis_flame_front.py          # Extração da frente de chama e curvas de decaimento do jato isotérmico
│   └── analysis_uncertainty.py          # Cálculo de autocorrelação temporal, correção do RMS e incertezas
│
├── .gitignore                           # Ficheiros e diretorias ignorados pelo controlo de versões
├── requirements.txt                     # Dependências de Python necessárias para correr o projeto
└── Readme.md                            # Documentação do projeto
```

---

## Metodologia e Equações Principais

### 1. Estatística de 1.ª e 2.ª Ordem (Média e RMS)
Para um conjunto de $N$ imagens, após subtração do sinal de fundo médio ($\bar{I}_{\text{bg}}$), a intensidade instantânea corrigida em cada píxel $(i,j)$ é dada por:
$$I(i,j,t) = I_{\text{raw}}(i,j,t) - \bar{I}_{\text{bg}}(i,j)$$

O campo médio ($\bar{I}$) e a flutuação quadrática média ($\text{RMS}$) são calculados como:
$$\bar{I}(i,j) = \frac{1}{N} \sum_{t=1}^{N} I(i,j,t)$$
$$I_{\text{rms}}(i,j) = \sqrt{\frac{1}{N-1} \sum_{t=1}^{N} \left( I(i,j,t) - \bar{I}(i,j) \right)^2}$$

### 2. Correção do Ruído de Fundo no RMS (`rms_*_corrigido.npy`)
Como a variância de variáveis independentes é aditiva, a flutuação medida ($I_{\text{rms, medido}}^2$) contém a contribuição da flutuação física real ($I_{\text{rms, real}}^2$) somada à variância do ruído eletrónico/ótico de fundo ($\sigma_{\text{bg}}^2$). O RMS corrigido é obtido por:
$$I_{\text{rms, corrigido}}(i,j) = \sqrt{\max\left(0, \, I_{\text{rms, medido}}^2(i,j) - \sigma_{\text{bg}}^2(i,j)\right)}$$

---

## Instalação e Como Executar

### 1. Pré-requisitos
Certifica-te de que tens o **Python 3.8+** instalado. Recomenda-se a utilização de um ambiente virtual (`venv` ou `conda`).

```bash
# Criar e ativar ambiente virtual (opcional mas recomendado)
python -m venv venv
source venv/bin/activate        # No macOS/Linux
# ou
venv\Scripts\activate           # No Windows

# Instalar as dependências necessárias
pip install -r requirements.txt
```

### 2. Fluxo de Execução dos Scripts
Caso pretendas processar novamente os dados brutos (colocados previamente em `data/raw/`), executa os scripts pela seguinte ordem a partir da raiz do projeto:

1. **Processamento das matrizes principais:**
   ```bash
   python src/process_flame.py
   python src/process_iso.py
   ```
2. **Visualização base dos campos Médio e RMS:**
   ```bash
   python src/visualize.py
   python src/visualize_iso.py
   ```
3. **Análises avançadas (Incertezas, *Clipping* e Frente de Chama):**
   ```bash
   python src/analysis_clipping_bias.py
   python src/analysis_uncertainty.py
   python src/analysis_flame_front.py
   ```

Todos os ficheiros `.npy` e gráficos `.png` serão guardados automaticamente na pasta `data/processed/`.