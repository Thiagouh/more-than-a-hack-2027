# Organização do repositório

Reorganização de 26/10/2026. Não inclui implementação de um novo modelo de posse.

| Local | Responsabilidade |
|---|---|
| `src/` | Código futuro; módulos e atribuições serão decididos na próxima reunião |
| `configs/` | Parâmetros explícitos e identificáveis dos experimentos |
| `notebooks/` | Exploração e tutoriais, com imports a partir da raiz |
| `starter/` | Baseline e loaders de referência, preservando os comandos existentes |
| `evaluation/` | Código oficial e referências públicas preservados; README com link atualizado |
| `docs/` | Documentação da competição e trabalho da equipe |
| `data/` | Destinos locais privados; apenas guias de preparação e resumos permitidos no Git |
| `outputs/` | Previsões, métricas e outros resultados locais; apenas guias permitidos no Git |

A divisão de trabalho e as subpastas propostas foram retiradas a pedido do usuário. Não há
responsáveis ou arquitetura definidos. O código oficial de avaliação permanece como referência.

## Movimentações

- Documentos da competição: raiz → `docs/`.
- Pesquisa técnica preservada em `PESQUISA-INICIAL.md`; plano de atribuições e roteiro antigos removidos.
- Três tutoriais: `starter/*.ipynb` → `notebooks/`.
- Vídeo `BAR-ATH_tactical-002.mp4`: raiz → `data/tactical/BAR-ATH_tactical.mp4`.
- Vídeo `BAR-ATH_broadcast-001.mp4`: raiz → `data/broadcast/BAR-ATH_broadcast.mp4`.
- Nove ZIPs: raiz → `data/archives/`, preservados.
- Sete assets tabulares dos ZIPs: extraídos para `data/tactical/` e `data/broadcast/`, nos nomes
  esperados por `games.asset_paths()`. CRC conferido durante a extração e SHA-256 registrado no
  manifesto local `data/organization-manifest.json`, que não é publicado.

Os vídeos foram movidos por renomeação no mesmo filesystem, com tamanho e inode preservados,
sem cópia de seus aproximadamente 19 GB. O FFprobe encontrou 199.319 frames declarados no tactical
e 164.487 no broadcast, ambos a 25 fps e 1920×1080. O tracking broadcast tem 164.488 registros:
essa diferença de um frame permanece como limitação a conferir antes de usar a fonte adicional.
Não houve correção de conteúdo ou alteração de timeline dos vídeos.

## Execução

Ative `.venv` e execute a partir da raiz:

```bash
source .venv/bin/activate
python starter/load_smart_tagging.py
python starter/baseline.py --out outputs/submissions/baselines
python evaluation/validate_submission.py outputs/submissions/baselines/baseline_smooth50.csv
python evaluation/run_evaluation.py --results outputs/submissions/baselines --out outputs/evaluation
```

Para uma instalação nova: `python -m pip install -r requirements-lock.txt`. O lock básico foi
obtido em Python 3.13. A compatibilidade com outras versões precisa ser conferida antes de assumir
que todas as versões fixadas estarão disponíveis nelas. Os requirements originais continuam na raiz.

Os comandos oficiais que usam `evaluation/results/` continuam funcionando. Usar `--out` e
`--results` mantém os novos resultados em `outputs/` e evita sobrescrever os baselines versionados.
Os dados também podem ficar em outro disco, configurando `MTAH_DATA_DIR`.

Para tutoriais: instale `requirements-starter.txt`, execute `jupyter notebook` e abra
`notebooks/03_first_submission.ipynb`. Os notebooks localizam a raiz tanto quando o kernel inicia
na raiz quanto em `notebooks/`; o validador usa o mesmo Python do kernel.

## Arquivos auxiliares locais

`graft/` é o mapa regenerável criado por `trail build`, não participa da solução. `.ignore` permite
pesquisar esse mapa com ripgrep. `.gitignore` exclui dados, ZIPs, `.venv`, caches e resultados.

Consulte o registro de validação ao final deste documento para saber quais verificações realmente
foram executadas. Os tutoriais completos dependem também de kloppy, matplotlib e Jupyter.

## Validação executada em 26/10/2026

- Ambiente básico `.venv` com Python 3.13: `pip check` passou; versões registradas em
  `requirements-lock.txt`, sem modificar bibliotecas globais.
- Os nove caminhos de assets esperados por `games.asset_paths()` existem.
- Alinhamento dos dois ATDs passou: IDs contíguos de 1 a 199319 (tactical) e 1 a 164488 (broadcast).
- Smart Tagging carregou 638 instâncias e passou a conferência de tempo nos dois kickoffs.
- Os quatro baselines foram regenerados em `outputs/submissions/baselines/` e comparados
  byte a byte com os quatro CSVs originais: todos idênticos.
- O validador oficial aprovou `baseline_smooth50.csv`, com 199.319 linhas.
- O scorer oficial avaliou os quatro arquivos e gerou `outputs/evaluation/scoreboard.csv` e
  relatórios. Os resultados públicos coincidiram com a tabela do README original. Isso confere
  o pipeline; não constitui seleção ou ajuste de modelos.
- Todos os arquivos de código, GT e CSV originais em `evaluation/` foram comparados com HEAD:
  sem alterações. Somente o link para a documentação em seu README foi atualizado.
- A comparação das árvores sintáticas confirmou a preservação da lógica executável de
  `games.py` e dos scripts de referência editados, descontando docstrings e o novo caminho
  informado pela mensagem de dados ausentes.
- Links locais de 22 arquivos Markdown e dos notebooks passaram. Todas as células de código
  dos três notebooks compilam; a localização da raiz foi exercitada a partir da raiz e de
  `notebooks/`. As células completas de exploração não foram executadas, pois o ambiente básico
  não instala as dependências opcionais de Jupyter, matplotlib e kloppy.
- Dados extraídos, vídeos, arquivos ZIP, ambiente e previsões foram conferidos como ignorados
  pelo Git. Nenhum dado da partida foi adicionado ao índice.
- Mapa de contexto atualizado com `trail build`.

Não houve commit, push ou envio de submissão nesta organização.

## Revisão dos guias

Removidas as atribuições e a estrutura por frentes. As explicações em português foram reunidas em [GUIA-PROJETO.md](GUIA-PROJETO.md), após a retirada
dos resumos dispersos para reduzir a quantidade de arquivos. A preparação individual dos dados está em [DADOS-LOCAIS.md](DADOS-LOCAIS.md). Somente guias e
diretórios de destino são destinados ao Git; cada participante obtém seu pacote autorizado.
