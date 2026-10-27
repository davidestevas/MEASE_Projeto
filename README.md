# Reconstrução Tomográfica de Jatos Reativos e Isotérmicos Axissimétricos via Schlieren e Inversão de Abel

## Identificação Académica e Contexto do Projeto

* **Unidade Curricular:** Métodos Experimentais para Análise de Sistemas Energéticos (MEASE) — 2.º Ano de Mestrado
* **Faculdade:** Instituto Superior Técnico
* **Tema do Trabalho:** Reconstrução Tomográfica 3D de Escoamentos Axissimétricos por Técnica de Schlieren e Transformada Inversa de Abel (*Onion-Peeling* / ART) e Análise da Comutatividade de Operadores Estatísticos
* **Grupo:** Grupo 22
* **Autores:** David Esteves (ist1116011), David Ferreira (ist 1107077) , Daniel Ascenso (ist 1113290)
* **Ano Letivo:** 2026/2027

### Breve Introdução
Este trabalho foi desenvolvido no âmbito da unidade curricular de **Métodos Experimentais para Análise de Sistemas Energéticos (MEASE)** e tem como objetivo a caracterização experimental e estatística de uma chama turbulenta estabilizada por um corpo rombo (*bluff-body*) em comparação com um jato isotérmico não-reativo (caso de controlo). Através do processamento de imagens de alta velocidade obtidas pela técnica ótica de **Schlieren**, aplica-se a **Transformada Inversa de Abel** discreta para desonvoluir a informação integrada na linha de visão e recuperar os campos radiais locais de gradiente de índice de refração, avaliando rigorosamente a comutatividade entre reconstrução tomográfica e operadores estatísticos (Média e RMS), o impacto de truncatura (*zero-clipping*), a propagação do ruído instrumental e o orçamento global de incertezas experimentais.

---

## 1. Contexto Físico e Grandeza Reconstruída ($\partial n / \partial z$)
Este repositório implementa a **Transformada Inversa de Abel** discreta através da **Técnica de Reconstrução Algébrica (ART / *Onion-Peeling*)** para reconstruir campos radiais planares a partir de registos **Schlieren** de alta velocidade ($400\text{ fps}$, $512 \times 512\text{ px}$) integrados na linha de visão:
1. **Chama turbulenta estabilizada por corpo rombo (`Chama`)** — $N = 3027$ imagens;
2. **Jato isotérmico não-reativo (`ISO`)** — $N = 3887$ imagens (caso de controlo / caso nulo);
3. **Referência de fundo (`Background`)** — $N = 717$ imagens.

### Orientação da Lâmina (*Knife-Edge*) e Paridade do Sinal
A partir da projeção subtraída do fundo $P(x, z) = I(x, z) - \bar{I}_{\text{bg}}(x, z)$, a recuperação da projeção média bruta ($\bar{P} = \bar{f} A^T$) demonstra que o sinal no eixo de simetria é não-nulo e simétrico: $\text{mediana}\, p(0) = -21.57\text{ u.a.} \approx \text{mediana}\, p(1..3) = -22.08\text{ u.a.} \neq 0$.  
Como o sinal projetado é **par** relativamente ao eixo de simetria:
* A lâmina Schlieren encontra-se orientada horizontalmente, sendo sensível ao **gradiente axial do índice de refração**;
* A matriz geométrica triangular superior do *Onion-Peeling* ($A^{-1}$) é o operador matemático exato para a inversão;
* A grandeza reconstruída $f(r, z) = A^{-1} P(x, z)$ corresponde ao **gradiente axial local do índice de refração $\partial n / \partial z$ em unidades arbitrárias (u.a.)**, e **não** diretamente à densidade absoluta do gás $\rho(r, z)$.
* Na chama reativa, o campo médio $\partial n / \partial z$ apresenta um núcleo interno positivo ($r < 55\text{ px}$) e uma banda térmica negativa pronunciada ($\text{FWHM mediana} = 29.0\text{ px}$, intervalo interdecil $23\text{--}37\text{ px}$) ancorada no bordo do corpo rombo (*bluff-body*, diâmetro $D_b \approx 116\text{ px}$, raio $R_b \approx 58\text{ px}$).

---

## 2. Comutatividade dos Operadores Estatísticos (`ab1` vs. `ab2`)
Em todos os ficheiros e scripts do repositório, os sufixos **`ab1`** e **`ab2`** designam a **Abordagem 1** e a **Abordagem 2**:

* **Abordagem 1 (`ab1`: Tomografia $\to$ Estatística)**: Cada frame instantâneo $P_k$ é reconstruído individualmente ($f_k = P_k (A^{-1})^T$) e a Média temporal e o RMS populacional (normalizado por $N$) são calculados no espaço tomográfico 3D:
  $$\bar{f}_{\text{ab1}}(z,r) = \frac{1}{N}\sum_{k=1}^{N} f_k(z,r), \qquad \text{RMS}_{\text{ab1}}(z,r) = \sqrt{\frac{1}{N}\sum_{k=1}^{N} f_k(z,r)^2 - \bar{f}_{\text{ab1}}(z,r)^2}$$
* **Abordagem 2 (`ab2`: Estatística $\to$ Tomografia)**: A Média e o RMS são calculados diretamente sobre as projeções 2D e multiplicados por $A^{-1}$ uma única vez no final.
  * **Comutatividade da Média**: Pela linearidade do operador inverso de Abel, as duas abordagens são numericamente idênticas:
    $$\frac{\|\bar{f}_{\text{ab1}} - \bar{f}_{\text{ab2}}\|_2}{\|\bar{f}_{\text{ab1}}\|_2} = 2.48 \times 10^{-13}$$
  * **Não-Comutatividade do RMS**: Como a raiz quadrada é um operador não-linear, a Abordagem 2 gera valores negativos fisicamente impossíveis ($\min = -1.4211\text{ u.a.}$ na chama), resultando num erro relativo de norma $L_2$ de:
    $$\frac{\|\text{RMS}_{\text{ab1}} - \text{RMS}_{\text{ab2}}\|_2}{\|\text{RMS}_{\text{ab1}}\|_2} = 83.5\% \text{ na Chama } (r = 0.50) \quad \text{e} \quad 94.6\% \text{ no ISO } (r = 0.25)$$

---

## 3. Estrutura do Repositório

```text
MEASE_Projeto22/
├── data/
│   ├── raw/                          # Imagens .tif originais (Background, Chama, ISO)
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
│       └── delta_eixo.npy            # Desvio sub-píxel medido do eixo [delta_medido, delta_aplicado]
├── src/
│   ├── utils.py                      # Matriz geométrica triangular superior A e inversa A^-1
│   ├── process_flame.py              # Medição do eixo (delta) + Abordagens 1 e 2 para a Chama
│   ├── process_iso.py                # Processamento das Abordagens 1 e 2 para o jato ISO
│   ├── analysis_uncertainty.py       # Comutatividade, ruído de fundo, ACF (N_eff) e orçamento de incerteza
│   ├── visualize.py                  # Mapas 2x2 da Chama (com máscara do corpo rombo Z >= 325 px)
│   ├── visualize_iso.py              # Mapas 2x2 do ISO na mesma escala de cor da Chama (Caso Nulo)
│   ├── analysis_flame_front.py       # Seguimento axial completo da frente de chama (z = 20..315 px)
│   └── analysis_clipping_bias.py     # Quantificação do viés severo induzido por zero-clipping
├── requirements.txt
└── Readme.md
```

---

## 4. Síntese dos Resultados Principais e Auditoria de Incertezas

1. **Medição do Eixo de Simetria ($\delta$)**: A minimização da assimetria par $|M(x_0 + r, z) - M(x_0 - r, z)|$ entre as duas metades da imagem média bruta ($512 \times 512$) identifica o eixo real de simetria na coluna $x_0 = 260.940\text{ px}$ ($\delta = +4.940\text{ px}$ face ao centro geométrico nominal $256$), sendo aplicada interpolação sub-píxel automática na extração da meia-secção reconstruída.
2. **Viés de Truncatura (*Zero-Clipping*)**: O corte de valores negativos após a subtração de fundo (`img_sub < 0 = 0`) viola a conservação do integral de Abel e destrói **$93.4\%$** da norma $L_2$ do campo médio no domínio do escoamento ($Z < 325\text{ px}$), reduzindo-a de $87.8934$ para $5.8431$ (retendo apenas $6.6\%$ da energia).
3. **Correção do Ruído Propagado e Caso Nulo ISO**:
   * O ruído bruto médio do sensor no fundo é $1.4356\text{ u.a.}$ Propagado analiticamente através de $\text{Var}(f_{\text{bg}}) = \text{Var}(P_{\text{bg}}) ((A^{-1})^T)^{\circ 2}$, o ruído reconstruído $\sigma_{\text{rec}}$ atinge $0.9187\text{ u.a.}$ no eixo ($R=0$, devido à singularidade $1/r$) e tem média radial de $0.0716\text{ u.a.}$
   * A subtração em quadratura $\text{RMS}_{\text{corr}} = \sqrt{\max(\text{RMS}_{\text{ab1}}^2 - \sigma_{\text{rec}}^2, 0)}$ corrige **$16.0\%$** do RMS da chama no eixo ($1.5769 \to 1.3308\text{ u.a.}$) e **$0.8\%$** na frente de chama ($0.4054 \to 0.4002\text{ u.a.}$).
   * No jato isotérmico (`ISO`), o rácio $\text{RMS}_{\text{ISO}} / \sigma_{\text{rec}}$ situa-se entre $0.89$ e $1.26$ na janela $r = 20\text{--}150\text{ px}$ (enquanto na frente de chama o sinal é $\sim 8.6\times$ superior ao ruído). Após a correção de ruído, o RMS ISO em $R=20\text{ px}$ cai de $0.131$ para $0.025\text{ u.a.}$, confirmando que o caso ISO constitui um **caso nulo** dominado pelo ruído instrumental.
4. **Deteção da Frente de Chama em Todas as Alturas**: O seguimento axial ($z = 20\text{--}315\text{ px}$) utiliza o **mínimo do campo médio** como critério principal e o **bordo exterior do patamar do RMS corrigido** como verificação, capturando a ancoragem da chama no bordo do corpo rombo e a sua expansão até à região cilíndrica a jusante.
5. **Orçamento Global de Incertezas (Campo Médio)**:
   * **Autocorrelação Temporal ($400\text{ fps}$)**: A partir de 3 sondas na zona reativa ($Z \in \{40, 80, 120\}\text{ px}$, $R=98\text{ px}$), o tempo integral de autocorrelação $\tau_{\text{int}} = \Delta t \left(\frac{1}{2} + \sum \rho(\tau)\right)$ é de $7.98\text{ ms}$, $9.10\text{ ms}$ e $7.95\text{ ms}$, resultando num número médio de amostras efetivamente independentes **$N_{\text{eff}} \approx 455$** (vs. $N = 3027$). O erro padrão estatístico (aplicável exclusivamente à Média) sobe de $1.39\%$ (ingénuo) para **$3.58\%$** (validado a $\sim 10\%$ por *Moving Block Bootstrap*).
   * **Incerteza Combinada**: Combinando em quadratura o erro estatístico da média ($\text{SE}_{\text{ACF}} = 3.58\%$), a sensibilidade a uma incerteza residual de $\pm 0.5\text{ px}$ no eixo para a largura real da banda ($W = 25\text{ px} \to 3.51\%$) e o erro algorítmico puro de discretização contínua-discreta sem *inverse crime* ($3.17\%$), obtém-se uma **incerteza relativa combinada realista de $5.93\%$** na zona ativa (e um limite pessimista condicional de **$10.95\%$** para $W = 8.8\text{ px}$).

---

## 5. Ordem Correta de Execução

Como os scripts de visualização e de frente de chama utilizam as matrizes de RMS corrigidas do ruído (`rms_ab1_corrigido.npy` e `rms_iso_ab1_corrigido.npy`), o script `analysis_uncertainty.py` deve ser executado imediatamente após o processamento base:

```bash
# 1. Processamento Tomográfico Base (Chama e ISO)
python src/process_flame.py
python src/process_iso.py

# 2. Auditoria de Incertezas e Geração dos Mapas de RMS Corrigido
python src/analysis_uncertainty.py

# 3. Geração dos Mapas 2D, Frente de Chama e Análise de Clipping
python src/visualize.py
python src/visualize_iso.py
python src/analysis_flame_front.py
python src/analysis_clipping_bias.py
```
