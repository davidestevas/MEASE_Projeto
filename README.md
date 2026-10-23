MEASE - Seeing the Invisible: Schlieren Tomography of Bluff-Body Flows

Disciplina: Métodos Experimentais para Análise de Sistemas Energéticos (MEASE)

Grupo: 22

Este repositório contém o código-fonte desenvolvido para o processamento de imagens do projeto laboratorial de MEASE. O objetivo é analisar o escoamento ao redor de um bluff-body utilizando a técnica ótica de Schlieren e reconstrução tomográfica axissimétrica (Onion-Peeling).

📌 Descrição do Projeto

Os ensaios decorreram com a captura de imagens a 400 fps, resolução espacial de $512 \times 512$ px e profundidade de 8-bits (monocromático).

Foram analisadas duas condições principais:

Caso Iso (Escoamento Isotérmico): 3887 imagens do escoamento a frio na esteira do bluff-body.

Caso Chama (Escoamento Reativo): 3027 imagens da chama pré-misturada ($\text{CH}_4$-ar) com $\phi \approx 1.25$.

Metodologia de Processamento

O processamento das imagens gera mapas do campo de densidades (Média e RMS) através de duas abordagens matemáticas comparativas:

Abordagem 1: A reconstrução tomográfica axissimétrica é aplicada a cada instante (frame a frame), seguida do cálculo estatístico da Média e do RMS.

Abordagem 2: A estatística (Média e RMS) é calculada sobre as projeções brutas, sendo a reconstrução tomográfica aplicada num único passo no final.

📂 Estrutura do Repositório

MEASE_Projeto22/
├── .gitignore
├── requirements.txt
├── README.md
├── data/
│   ├── raw/                 # DADOS BRUTOS (NÃO SINCRONIZADOS COM O GIT)
│   │   ├── Background/tiff/ # 717 imagens
│   │   ├── Chama/           # 3027 imagens
│   │   └── ISO/             # 3887 imagens
│   └── processed/           # Resultados gerados (matrizes .npy e gráficos .png)
├── docs/                    # Relatório final em LaTeX
└── src/                     # Código-fonte Python
    ├── utils.py             # Funções de matriz geométrica (ART / Onion-Peeling)
    ├── approach1.py         # Processamento do caso Chama (Abordagens 1 e 2 simultâneas)
    ├── approach2.py         # Processamento do caso ISO (Abordagens 1 e 2 simultâneas)
    ├── visualize.py         # Geração dos mapas de cores da Chama
    └── visualize_iso.py     # Geração dos mapas de cores do ISO


⚙️ Configuração Inicial (Para Membros do Grupo)

Para poderes correr este código no teu computador, segue estes passos rigorosamente:

Clonar o repositório:

git clone https://github.com/davidestevas/MEASE_Projeto.git
cd MEASE_Projeto


Instalar dependências:
Garante que tens o Python instalado e executa:

pip install numpy imageio matplotlib scipy


Configurar os Dados (MUITO IMPORTANTE):
Devido ao tamanho elevado (vários GB), as imagens originais .tif estão ignoradas no .gitignore.

Descarrega a pasta "Group 22" do SharePoint da cadeira.

Coloca as pastas originais exatamente dentro do diretório data/raw/, respeitando a estrutura indicada acima. Sem isto, os scripts vão devolver um erro FileNotFoundError.

🚀 Como Executar o Código

Os scripts estão desenhados com caminhos relativos robustos (os.path), pelo que podes executá-los a partir de qualquer diretório.

1. Processar dados da Chama (gera matrizes .npy):

python src/approach1.py


2. Visualizar e gravar gráficos da Chama (.png):

python src/visualize.py


3. Processar dados Isotérmicos (ISO):

python src/approach2.py


4. Visualizar e gravar gráficos do ISO (.png):

python src/visualize_iso.py


📝 Autores

David Estevas (e restantes elementos do Grupo 22)
