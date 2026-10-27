# Reconstrução Tomográfica de Jatos Reativos e Isotérmicos Axissimétricos via Schlieren e Inversão de Abel

## Identificação Académica e Contexto do Projeto

- **Unidade Curricular:** Métodos Experimentais para Análise de Sistemas Energéticos (MEASE) — 2.º Ano de Mestrado em Engenharia Mecânica
- **Faculdade:** Instituto Superior Técnico (IST) — Universidade de Lisboa
- **Tema do Trabalho:** Reconstrução Tomográfica 3D de Escoamentos Axissimétricos por Técnica de Schlieren e Transformada Inversa de Abel (*Onion-Peeling* / ART) e Análise da Comutatividade de Operadores Estatísticos
- **Grupo:** Grupo 22
- **Autores:** David Esteves (ist1116011), David Ferreira (ist1107077), Daniel Ascenso (ist1113290)
- **Ano Letivo:** 2025/2026

### Breve Introdução
Este trabalho foi desenvolvido no âmbito da unidade curricular de **Métodos Experimentais para Análise de Sistemas Energéticos (MEASE)** e tem como objetivo a caracterização experimental e estatística de uma chama turbulenta estabilizada por um corpo rombo (*bluff-body*) em comparação com um jato isotérmico não-reativo (caso de controlo). Através do processamento de imagens de alta velocidade obtidas pela técnica ótica de **Schlieren**, aplica-se a **Transformada Inversa de Abel** discreta para desonvoluir a informação integrada na linha de visão e recuperar os campos radiais locais de gradiente de índice de refração, avaliando rigorosamente a comutatividade entre reconstrução tomográfica e operadores estatísticos (Média e RMS), o impacto de truncatura (*zero-clipping*), a propagação do ruído instrumental e o orçamento global de incertezas experimentais.

---

## 1. Introdução e Grandeza Física Reconstruída ($\partial n / \partial z$)
Este projeto desenvolve e valida um *pipeline* numérico de reconstrução tomográfica planar baseado na **Transformada Inversa de Abel** discreta — implementada através da **Técnica de Reconstrução Algébrica (ART / *Onion-Peeling*)** — aplicado a registos **Schlieren** de alta velocidade ($f_s = 400\text{ fps}$, resolução espacial de $512 \times 512\text{ px}$) de três conjuntos experimentais:
1. **Chama turbulenta estabilizada por corpo rombo (`Chama`)** — $N = 3027$ imagens instantâneas;
2. **Jato isotérmico não-reativo (`ISO`)** — $N = 3887$ imagens instantâneas (caso de controlo / caso nulo);
3. **Referência óptica de fundo (`Background`)** — $N = 717$ imagens sem escoamento.

### Orientação da Lâmina (*Knife-Edge*), Paridade e Definição Operacional da Frente de Chama
A partir da projeção subtraída do fundo médio $P(x, z) = I(x, z) - \bar{I}_{\text{bg}}(x, z)$, a recuperação da projeção média bruta ($\bar{P} = \bar{f} A^T$) demonstra que o sinal projetado no eixo de simetria é não-nulo e simétrico:
$$\text{mediana}\, p(0) = -21.74\text{ u.a.} \approx \text{mediana}\, p(1..3) = -22.17\text{ u.a.} \neq 0$$
Como o sinal projetado é uma função **par** relativamente ao eixo de simetria:
- A lâmina Schlieren (*knife-edge*) interceta os raios defletidos na direção axial, medindo a componente axial da refração. Consequentemente, a matriz triangular superior padrão do *Onion-Peeling* ($A^{-1}$) é o operador matemático exato da inversão, mas o campo reconstruído $f(r, z) = A^{-1} P(x, z)$ representa o **gradiente axial local do índice de refração ($\partial n / \partial z$) em unidades arbitrárias (u.a.)**, e **not** a densidade absoluta $\rho(r, z)$.
- Uma vez que a montagem mede apenas $\partial n / \partial z$, uma frente térmica perfeitamente paralela ao eixo produziria gradiente axial nulo. Assim, a localização da frente de chama é estabelecida através de uma **definição operacional dupla**:
  1. **Critério Principal (Centro da Banda Térmica):** o extremo (mínimo) de $\partial n / \partial z$ no campo médio ($r \approx 81\text{--}85\text{ px}$ na zona desenvolvida);
  2. **Critério Independente de Verificação (Bordo Exterior de Flutuações):** o raio exterior onde o perfil de $\text{RMS}$ corrigido do ruído decai para $50\%$ do seu patamar máximo na camada de corte ($r \approx 95\text{--}103\text{ px}$ com o eixo deslocado de $\delta = +5.28\text{ px}$, equivalente a $r \approx 101\text{--}108\text{ px}$ em relação à coluna nominal 256).

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
│   ├── raw/                        # Imagens .tif brutas (Background, Chama, ISO)
│   └── processed/                  # Matrizes .npy reconstruídas e figuras .png
│       ├── media_ab1.npy           # Campo médio da Chama - Abordagem 1
│       ├── media_ab2.npy           # Campo médio da Chama - Abordagem 2
│       ├── rms_ab1.npy             # RMS bruto da Chama - Abordagem 1
│       ├── rms_ab1_corrigido.npy   # RMS da Chama corrigido do ruído de fundo - Abordagem 1
│       ├── rms_ab2.npy             # RMS da Chama - Abordagem 2 (não-físico)
│       ├── media_iso_ab1.npy       # Campo médio ISO - Abordagem 1
│       ├── media_iso_ab2.npy       # Campo médio ISO - Abordagem 2
│       ├── rms_iso_ab1.npy         # RMS bruto ISO - Abordagem 1
│       ├── rms_iso_ab1_corrigido.npy # RMS ISO corrigido do ruído de fundo - Abordagem 1
│       ├── rms_iso_ab2.npy         # RMS ISO - Abordagem 2
│       ├── sigma_rec_bg.npy        # Mapa do ruído do sensor propagado por A^-1
│       ├── media_raw_full.npy      # Imagem média bruta 512x512 da Chama (ambas as metades)
│       └── delta_eixo.npy          # Parâmetros medidos do eixo de simetria (delta e dispersão)
├── src/
│   ├── utils.py                    # Matriz A^-1, extração sub-píxel (delta) e geometria do queimador
│   ├── process_flame.py            # Medição do eixo (janela +-15 px) + Abordagens 1 e 2 (Chama)
│   ├── process_iso.py              # Abordagens 1 e 2 para o jato ISO (com o mesmo delta)
│   ├── analysis_uncertainty.py     # Comutatividade, ruído, ACF (N_eff) e orçamento de incerteza
│   ├── visualize.py                # Mapas 2x2 da Chama (máscara do corpo rombo Z >= 325 px)
│   ├── visualize_iso.py            # Mapas 2x2 do ISO na mesma escala de cor da Chama (Caso Nulo)
│   ├── analysis_flame_front.py     # Seguimento axial completo (Mínimo da Média + Bordo 50% RMS)
│   └── analysis_clipping_bias.py   # Quantificação do viés severo induzido por zero-clipping
├── requirements.txt
└── Readme.md
