# Reconstrução Tomográfica de Jatos Reativos e Isotérmicos Axissimétricos via Schlieren e Inversão de Abel

## Identificação Académica e Contexto do Projeto

- **Unidade Curricular:** Métodos Experimentais para Análise de Sistemas Energéticos (MEASE) — 2.º Ano de Mestrado em Engenharia Mecânica
- **Faculdade:** Instituto Superior Técnico (IST) — Universidade de Lisboa
- **Tema do Trabalho:** Reconstrução Tomográfica 3D de Escoamentos Axissimétricos por Técnica de Schlieren e Transformada Inversa de Abel (*Onion-Peeling* / ART) e Análise da Comutatividade de Operadores Estatísticos
- **Grupo:** Grupo 22
- **Autores:** David Esteves (ist1116011), David Ferreira (ist1107077), Daniel Ascenso (ist1113290)
- **Ano Letivo:** 2026/2027

### Breve Introdução
Este trabalho foi desenvolvido no âmbito da unidade curricular de **Métodos Experimentais para Análise de Sistemas Energéticos (MEASE)** e tem como objetivo a caracterização experimental e estatística de uma chama turbulenta estabilizada por um corpo rombo (*bluff-body*) em comparação com um jato isotérmico não-reativo (caso de controlo). Através do processamento de imagens de alta velocidade obtidas pela técnica ótica de **Schlieren**, aplica-se a **Transformada Inversa de Abel** discreta para desconvoluir a informação integrada na linha de visão e recuperar os campos radiais locais de gradiente de índice de refração, avaliando rigorosamente a comutatividade entre reconstrução tomográfica e operadores estatísticos (Média e RMS), o impacto de truncatura (*zero-clipping*), a propagação do ruído instrumental e o orçamento global de incertezas experimentais.

---

## 1. Grandeza Física Reconstruída ($\partial n / \partial z$), Discretização Radial ($r = j + 0.5$) e Normalização ($r / R_b$)
Este projeto desenvolve e valida um *pipeline* numérico de reconstrução tomográfica planar baseado na **Transformada Inversa de Abel** discreta — implementada através da **Técnica de Reconstrução Algébrica (ART / *Onion-Peeling*)** — aplicado a registos **Schlieren** de alta velocidade ($f_s = 400\text{ fps}$, resolução espacial de $512 \times 512\text{ px}$) de três conjuntos experimentais:
1. **Chama turbulenta estabilizada por corpo rombo (`Chama`)** — $N = 3027$ imagens instantâneas;
2. **Jato isotérmico não-reativo (`ISO`)** — $N = 3887$ imagens instantâneas (caso de controlo / caso nulo);
3. **Referência ótica de fundo (`Background`)** — $N = 717$ imagens sem escoamento.

### Índice do Anel ($j$) vs. Raio Físico ($r = j + 0.5\text{ px}$) e Escala Adimensional ($r / R_b$)
Na discretização geométrica do *Onion-Peeling*, cada índice de coluna reconstruída $j \in \{0, \dots, 255\}$ representa o anel circular compreendido entre os raios $r_j = j$ e $r_{j+1} = j + 1$, pelo que o valor reconstruído $f_j$ corresponde fisicamente ao centro geométrico do anel **$r = j + 0.5\text{ px}$**.  
Para garantir que toda a caracterização geométrica é autofundamentada nos dados e independente de calibrações externas em milímetros, adota-se a **normalização adimensional $r / R_b$** pelo raio do corpo rombo medido diretamente na imagem de fundo ($R_b = 54.06\text{ px}$, $D_b = 2 R_b = 108.13\text{ px}$).

### Orientação da Lâmina (*Knife-Edge*), Paridade e Definição Operacional da Frente de Chama
A partir da projeção subtraída do fundo médio $P(x, z) = I(x, z) - \bar{I}_{\text{bg}}(x, z)$, a recuperação da projeção média bruta ($\bar{P} = \bar{f} A^T$) demonstra que o sinal projetado no eixo de simetria é não-nulo e simétrico:
$$\text{mediana}\, p(0) = -21.74\text{ u.a.} \approx \text{mediana}\, p(1..3) = -22.17\text{ u.a.} \neq 0$$
Como o sinal projetado é uma função **par** relativamente ao eixo de simetria:
- A lâmina Schlieren (*knife-edge*) interceta os raios defletidos na direção axial, medindo exclusivamente a componente axial da refração. Consequentemente, a matriz triangular superior padrão do *Onion-Peeling* ($A^{-1}$) é o operador matemático exato da inversão, mas o campo reconstruído $f(r, z) = A^{-1} P(x, z)$ representa o **gradiente axial local do índice de refração ($\partial n / \partial z$) em unidades arbitrárias (u.a.)**, e **não** a densidade absoluta $\rho(r, z)$.
- Uma vez que a montagem mede apenas $\partial n / \partial z$, uma frente térmica perfeitamente paralela ao eixo produziria gradiente axial nulo. Assim, a localização da frente de chama é estabelecida através de uma **definição operacional dupla** (para $z = 40\text{--}200\text{ px}$):
  1. **Critério Principal (Centro da Banda Térmica):** o extremo (mínimo) de $\partial n / \partial z$ no campo médio, localizado nos índices $j = 81\text{--}85$, correspondente ao raio físico **$r = 81.5\text{--}85.5\text{ px}$** e à posição adimensional **$r / R_b \approx 1.51\text{--}1.58$** (ancorado no bordo do corpo rombo $r / R_b = 1.00$ em $z \approx 320\text{ px}$);
  2. **Critério Independente de Verificação (Bordo Exterior de Flutuações):** o raio exterior onde o perfil de $\text{RMS}$ corrigido do ruído decai para $50\%$ do seu patamar máximo na camada de corte, localizado nos índices $j = 95\text{--}103$, correspondente ao raio físico **$r = 95.5\text{--}103.5\text{ px}$** e à posição adimensional **$r / R_b \approx 1.77\text{--}1.91$**.

---

## 2. Comutatividade dos Operadores Estatísticos (`ab1` vs. `ab2`)
Em todos os ficheiros e scripts do repositório, **`ab1`** e **`ab2`** designam a **Abordagem 1** e a **Abordagem 2** (Tópico 2 do projeto):

- **Abordagem 1 (`ab1`: Tomografia $\to$ Estatística):** Cada projeção instantânea $P_k$ é invertida primeiro ($f_k = P_k (A^{-1})^T$), calculando-se a Média temporal e o RMS populacional (normalizado por $N$) sobre as fatias radiais reconstruídas:
  $$\bar{f}_{\text{ab1}}(z,r) = \frac{1}{N}\sum_{k=1}^{N} f_k(z,r), \qquad \text{RMS}_{\text{ab1}}(z,r) = \sqrt{\frac{1}{N}\sum_{k=1}^{N} f_k(z,r)^2 - \bar{f}_{\text{ab1}}(z,r)^2}$$
- **Abordagem 2 (`ab2`: Estatística $\to$ Tomografia):** A Média e o RMS temporal são calculados sobre as projeções 2D integradas na linha de visão e multiplicados por $A^{-1}$ uma única vez no final.
  * **A Média Comuta:** Pela linearidade da transformação inversa de Abel, ambas as abordagens coincidem até à precisão de máquina:
    $$\frac{\Vert{}\bar{f}_{\text{ab1}} - \bar{f}_{\text{ab2}}\Vert{}_2}{\Vert{}\bar{f}_{\text{ab1}}\Vert{}_2} = 2.62 \times 10^{-13}$$
  * **O RMS Não Comuta:** Devido à não-linearidade do operador quadrático/raiz, a Abordagem 2 gera valores negativos fisicamente impossíveis ($\min = -1.1407\text{ u.a.}$ na chama), originando uma discrepância relativa de norma $L_2$ de:
    $$\frac{\Vert{}\text{RMS}_{\text{ab1}} - \text{RMS}_{\text{ab2}}\Vert{}_2}{\Vert{}\text{RMS}_{\text{ab1}}\Vert{}_2} = 82.3\% \text{ na Chama } (r = 0.52) \quad \text{e} \quad 93.6\% \text{ no ISO } (r = 0.28)$$

---

## 3. Estrutura do Repositório

```text
MEASE_Projeto22/
├── data/
│   ├── raw/                          # Imagens .tif brutas (Background, Chama, ISO)
│   └── processed/                    # Matrizes .npy reconstruídas e figuras .png
│       ├── media_ab1.npy             # Campo médio da Chama - Abordagem 1
│       ├── media_ab2.npy             # Campo médio da Chama - Abordagem 2
│       ├── rms_ab1.npy               # RMS bruto da Chama - Abordagem 1
│       ├── rms_ab1_corrigido.npy     # RMS da Chama corrigido do ruído de fundo - Abordagem 1
│       ├── rms_ab2.npy               # RMS da Chama - Abordagem 2 (não-físico)
│       ├── media_iso_ab1.npy         # Campo médio ISO - Abordagem 1
│       ├── media_iso_ab2.npy         # Campo médio ISO - Abordagem 2
│       ├── rms_iso_ab1.npy           # RMS bruto ISO - Abordagem 1
│       ├── rms_iso_ab1_corrigido.npy # RMS ISO corrigido do ruído de fundo - Abordagem 1
│       ├── rms_iso_ab2.npy           # RMS ISO - Abordagem 2
│       ├── sigma_rec_bg.npy          # Mapa do ruído do sensor propagado por A^-1
│       ├── media_raw_full.npy        # Imagem média bruta 512x512 da Chama (ambas as metades)
│       ├── delta_eixo.npy            # Parâmetros medidos do eixo de simetria (delta e dispersão)
│       └── geometria_queimador.npy   # Geometria medida no Background (D_b, R_b, centro_bg)
├── src/
│   ├── utils.py                      # Matriz A^-1, extração sub-píxel (delta) e geometria do queimador
│   ├── process_flame.py              # Medição do eixo (janela +-15 px) + Abordagens 1 e 2 (Chama)
│   ├── process_iso.py                # Abordagens 1 e 2 para o jato ISO (com o mesmo delta)
│   ├── analysis_uncertainty.py       # Comutatividade, ruído, ACF SE(z) e orçamento de incerteza
│   ├── visualize.py                  # Mapas 2x2 da Chama (máscara do corpo rombo Z >= 325 px)
│   ├── visualize_iso.py              # Mapas 2x2 do ISO na mesma escala de cor da Chama (Caso Nulo)
│   ├── analysis_flame_front.py       # Seguimento axial completo (r = j + 0.5 px e r / R_b)
│   └── analysis_clipping_bias.py     # Quantificação do viés severo induzido por zero-clipping
├── requirements.txt
└── Readme.md
```

---

## 4. Síntese dos Resultados Principais e Orçamento de Incertezas

1. **Determinação Empírica do Eixo de Simetria ($\delta$) e Geometria Adimensional ($r / R_b$):**
   * A minimização da assimetria par $|M(x_0 + r, z) - M(x_0 - r, z)|$ numa janela alargada de $\pm 15\text{ px}$ sobre a imagem média bruta ($512 \times 512$) localiza o eixo real de simetria em **$\delta = +5.28\text{ px}$** (coluna $261.28$), com região a 10% do mínimo em $[+4.70, +5.64]\text{ px}$. Por blocos axiais de $z$, o ótimo varia entre $+4.63\text{ e }+5.39\text{ px}$, apresentando uma dispersão de **$\pm 0.39\text{ px}$** ($\approx 0.4\text{ px}$), o que fundamenta empiricamente a incerteza residual de $\pm 0.5\text{ px}$ adotada no orçamento de incerteza.
   * **Validação Independente no Background:** A deteção direta dos bordos do queimador na imagem de fundo ($Z = 335\text{--}500\text{ px}$, com refinamento parabólico sub-píxel) mede um diâmetro do corpo rombo de **$D_b = 108.13\text{ px}$** (raio **$R_b = 54.06\text{ px}$**) centrado na coluna **$260.85\text{ px}$** ($\delta_{\text{bg}} = +4.85\text{ px}$), validando de forma independente o desvio do eixo ($\Delta \delta = 0.43\text{ px}$ face a $+5.28\text{ px}$). Usando $R_b = 54.06\text{ px}$ como escala característica, a largura mediana da banda negativa ($\text{FWHM} = 30.0\text{ px}$, intervalo $10\text{--}90\%$ de $25\text{--}37\text{ px}$) corresponde a **$\text{FWHM} / R_b \approx 0.55$** ($0.46\text{--}0.68$).
2. **Viés de Truncatura (*Zero-Clipping*):** Com o deslocamento $\delta = +5.28\text{ px}$ aplicado consistentemente, truncar os valores negativos após a subtração de fundo (`img_sub < 0 = 0`) destrói **$92.9\%$** da norma $L_2$ do campo médio no domínio do escoamento ($Z < 325\text{ px}$), reduzindo-a de $87.5204$ para $6.1986$ (retendo apenas $7.1\%$).
3. **Propagação de Ruído e Caso Nulo ISO:**
   * A propagação analítica da variância do fundo ($\text{Var}(f_{\text{bg}}) = \text{Var}(P_{\text{bg}}) ((A^{-1})^T)^{\circ 2}$) resulta num ruído reconstruído $\sigma_{\text{rec}} = 0.7858\text{ u.a.}$ no anel do eixo ($j=0$, $r=0.5\text{ px}$) e $0.0598\text{ u.a.}$ em média radial.
   * A correção em quadratura $\text{RMS}_{\text{corr}} = \sqrt{\max(\text{RMS}_{\text{ab1}}^2 - \sigma_{\text{rec}}^2, 0)}$ reduz o RMS da chama em **$-15.0\%$** no eixo ($1.3675 \to 1.1624\text{ u.a.}$) e em **$-1.3\%$** na frente de chama ($0.4225 \to 0.4170\text{ u.a.}$ em $Z=80, j=85, r=85.5\text{ px}$).
   * No jato isotérmico (`ISO`), $\text{RMS}_{\text{ISO}} / \sigma_{\text{rec}} \in [0.75, 1.23]$ para $j = 20\text{--}150$, com **$79.7\%$** dos píxeis sem sinal resolvível acima do ruído (enquanto na frente de chama o sinal é $\sim 10.3\times$ o ruído). Após a subtração de $\sigma_{\text{rec}}^2$, o RMS ISO em $j=20$ ($r=20.5\text{ px}$) cai de $0.104$ para $0.015\text{ u.a.}$ e a meia-largura aparente colapsa de $44.5\text{--}56.5\text{ px}$ para $20.5\text{--}23.5\text{ px}$, confirmando o caso ISO como **caso nulo**.
4. **Orçamento Global de Incertezas (Aplicável Exclusivamente ao Campo Médio):**
   * **Erro Padrão Estatístico em Função da Altura $\text{SE}(z)$:** Avaliado a partir de 3 sondas temporais ($400\text{ fps}$) no centro da banda reativa ($j = 82$, $r = 82.5\text{ px}$, $r / R_b = 1.53$), o número efetivo de amostras independentes varia cerca de $4\times$ com a altura axial: **$N_{\text{eff}} = 174$** ($\tau_{\text{int}} = 21.80\text{ ms}$) em $Z=40\text{ px}$, **$N_{\text{eff}} = 274$** ($\tau_{\text{int}} = 13.83\text{ ms}$) em $Z=80\text{ px}$ e **$N_{\text{eff}} = 692$** ($\tau_{\text{int}} = 5.47\text{ ms}$) em $Z=120\text{ px}$. Isto traduz-se num erro padrão relativo mediano na zona ativa de **$5.54\%$**, **$4.41\%$** e **$2.77\%$**, respetivamente (vs. $1.33\%$ ingénuo e $3.74\%$ usando a média $N_{\text{eff}} = 380$). O estimador por ACF constitui um limite superior conservador, situando-se o *Moving Block Bootstrap* entre **$9\%\text{ e }32\%$** abaixo da ACF.
   * **Erro do Eixo ($\pm 0.5\text{ px}$) e Erro Algorítmico (sem *inverse crime*):** Utilizando a largura real medida da banda negativa (**$\text{FWHM mediana} = 30.0\text{ px}$**, $\text{FWHM}/R_b \approx 0.55$), um desvio residual de $\pm 0.5\text{ px}$ induz um erro $L_2$ de **$2.95\%$**. O erro puro de discretização do algoritmo *Onion-Peeling* face ao integral contínuo de Abel (sem *inverse crime*, avaliado no centro geométrico dos anéis $r = j + 0.5\text{ px}$) é de **$0.35\%$**.
   * **Incerteza Combinada em Quadratura:** Combinando $\text{SE}(z)$, erro do eixo ($2.95\%$) e erro algorítmico ($0.35\%$), a incerteza relativa combinada realista do campo médio situa-se no intervalo de **$4.1\%\text{ a }6.3\%$ ($\approx 4\text{--}6\%$)** em função da altura axial ($4.06\%$ em $Z=120\text{ px}$, $5.32\%$ em $Z=80\text{ px}$ e $6.28\%$ em $Z=40\text{ px}$; ou **$4.78\%$** tomando a referência média $N_{\text{eff}}=380$), e num limite pessimista condicional de **$10.55\%$** para uma camada fina $W = 8.8\text{ px}$.

---

## 5. Ordem Correta de Execução dos Scripts

O script `analysis_uncertainty.py` **tem obrigatoriamente de ser executado antes** dos scripts de visualização (`visualize.py`, `visualize_iso.py`) e de seguimento da frente de chama (`analysis_flame_front.py`), uma vez que gera e grava as matrizes de RMS corrigidas do ruído (`rms_ab1_corrigido.npy` e `rms_iso_ab1_corrigido.npy`) e a geometria medida do queimador (`geometria_queimador.npy`):

**Sequência obrigatória:**  
`process_flame.py` $\to$ `process_iso.py` $\to$ `analysis_uncertainty.py` $\to$ `visualize.py` $\to$ `visualize_iso.py` $\to$ `analysis_flame_front.py` $\to$ `analysis_clipping_bias.py`

```bash
python src/process_flame.py
python src/process_iso.py
python src/analysis_uncertainty.py
python src/visualize.py
python src/visualize_iso.py
python src/analysis_flame_front.py
python src/analysis_clipping_bias.py
```
