# Guia do projeto em português

Este repositório fornece ferramentas para estimar **posse de bola por frame** de uma partida.
A arquitetura da solução e a divisão de trabalho ainda serão discutidas pela equipe.

## O que entra e o que sai

Entram vídeo, tracking de jogadores/bola, eventos automáticos e homografias. A saída é um CSV
`frame,label`, com 199.319 linhas de dados na timeline da câmera tática:

| Rótulo | Significado |
|---|---|
| H | Mandante em controle, bola viva |
| A | Visitante em controle, bola viva |
| D | Bola fora de jogo |
| X | Abstenção |

Uma mudança de posse exige controle do adversário; um toque ou desvio isolado não basta.
Bola ausente no tracking não significa automaticamente bola parada. Os sinais fornecidos são
automáticos e têm lacunas; slots de tracking não são identidades permanentes de jogadores.

## Pastas

| Pasta | Conteúdo |
|---|---|
| `docs/` | Documentação original da competição, pesquisa e guias locais |
| `starter/` | Baseline, loaders e visualização fornecidos como referência |
| `evaluation/` | Código oficial de avaliação, GT público e quatro previsões baseline |
| `notebooks/` | Três tutoriais de exploração e primeira submissão |
| `data/` | Destinos para os arquivos que cada participante obtém individualmente |
| `src/` | Espaço reservado para implementação futura, sem módulos definidos |
| `configs/` | Espaço para configurações que acompanhem implementações reais |
| `scripts/` | Conferência local da presença dos arquivos de dados |
| `outputs/` | Previsões e relatórios locais, ignorados pelo Git |
| `graft/` | Mapa auxiliar regenerável do código, ignorado pelo Git |

`.git/` contém metadados do controle de versão; `.venv/` é o ambiente Python; `__pycache__/`
contém cache do interpretador. Não são materiais do desafio.

## Documentos originais em inglês

| Documento | O que você precisa absorver |
|---|---|
| [README.md](../README.md) | Visão geral, comandos iniciais, recursos e calendário |
| [CHALLENGE.md](CHALLENGE.md) | Definição de controle, mudanças de posse e H/A/D/X. H→D→H são duas posses; D não é posse |
| [DATA.md](DATA.md) | Formatos, coordenadas, qualidade e alinhamento dos dados. Tactical é a referência; broadcast não vem sincronizado |
| [EVALUATION.md](EVALUATION.md) | spellF1 compara posses completas por IoU temporal; macroF1 avalia H/A/D por frame e é o primeiro desempate |
| [SUBMISSION.md](SUBMISSION.md) | CSV completo válido, repositório público que o regenera exatamente e documento técnico curto |
| [RULES.md](RULES.md) | Métodos devem ser inspecionáveis e explicados; dados privados não podem ser redistribuídos. A cópia local antecede esclarecimentos oficiais |
| [FAQ.md](FAQ.md) | Bola ausente, abstenção, arquivos inválidos, broadcast, ferramentas permitidas e dúvidas de execução |
| [OVERVIEW.md](OVERVIEW.md) | Contexto da competição e vocabulário de fases, transições, reposições e zonas de jogo; não substitui as especificações |
| [data/README.md](../data/README.md) | Onde colocar o pacote e como usar `MTAH_DATA_DIR`; o scorer funciona sem esses assets quando avalia os CSVs já fornecidos |
| [evaluation/results/README.md](../evaluation/results/README.md) | Como avaliar os CSVs da pasta; as quatro respostas baseline são públicas, mas novas previsões permanecem locais |

**Avaliação:** o scorer pareia posses uma a uma e calcula F1 em diferentes limiares de
sobreposição. Acertar muitos frames não compensa necessariamente fragmentar uma posse em várias.
Em frames avaliados, macroF1 cobra X como erro; nas métricas temporais, o scorer preenche lacunas
dentro dos trechos válidos. Assim, abster-se também pode prolongar uma equipe errada numa troca.

**Referência pública:** cobre somente cinco minutos. Serve para conferir o pipeline; as
[regras oficiais atuais](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md)
não permitem usá-la para treino ou ajuste, nem produzir rótulos/correções manuais desta partida.
Dados de outros jogos podem apoiar o desenvolvimento dentro dos termos. O GT completo é privado.

**Entrega:** cabeçalho `frame,label`, frames 0–199318 em ordem, rótulos válidos e sem células
vazias. O validador confere formato; não comprova acurácia. O formulário é enviado aos inscritos,
sem leaderboard ao vivo. O prazo indicado no kit é 17/11/2026; confirme horário/fuso no formulário.

## Código e arquivos de referência

| Arquivo | Função |
|---|---|
| [games.py](../games.py) | Caminhos, fontes, FPS, dimensões, contagens e conversões de contadores |
| [starter/baseline.py](../starter/baseline.py) | Jogador mais próximo da bola, votação temporal e exportação dos quatro CSVs |
| [starter/load_atd.py](../starter/load_atd.py) | Leitura e conferência do tracking; `check_alignment_raw()` evita carregar toda a tabela |
| [starter/load_smart_tagging.py](../starter/load_smart_tagging.py) | XML de eventos → intervalos e tags; confere a base de tempo nos kickoffs |
| [starter/load_homography.py](../starter/load_homography.py) | Projeção campo↔pixels, validade e alinhamento das matrizes |
| [starter/utils.py](../starter/utils.py) | Desenho do campo, frames e animações do tracking |
| [starter/make_dummy_submission.py](../starter/make_dummy_submission.py) | CSVs de resposta constante para conferir o fluxo sem usar a partida |
| [evaluation/possession_metrics.py](../evaluation/possession_metrics.py) | Implementação das métricas oficiais e de sua validação |
| [evaluation/run_evaluation.py](../evaluation/run_evaluation.py) | Lê GT e previsões, calcula notas e gera scoreboard/relatórios |
| [evaluation/validate_submission.py](../evaluation/validate_submission.py) | Confere linhas, frames, rótulos e vazios do CSV |
| [scripts/verificar_dados.py](../scripts/verificar_dados.py) | Confere arquivos locais existentes e não vazios; não baixa nem altera dados |

Em `evaluation/gt/`, `BAR-ATH_possession_gt_public.csv` contém a referência pública e o JSON
adjacente descreve sua janela e alinhamento. Os arquivos `baseline_raw.csv` e
`baseline_smooth25/50/125.csv`, em `evaluation/results/`, são previsões do mesmo algoritmo com
diferentes janelas de suavização — não são respostas verdadeiras. Preserve os originais e use
`--out`/`--results` para experimentar em `outputs/`.

## Tutoriais

1. [01_smart_tagging.ipynb](../notebooks/01_smart_tagging.ipynb): eventos, códigos, tags, N/A e tempos.
2. [02_tracking_atd.ipynb](../notebooks/02_tracking_atd.ipynb): posições, coordenadas, lacunas e visualização.
3. [03_first_submission.ipynb](../notebooks/03_first_submission.ipynb): tracking → baseline → CSV → avaliação.

Os notebooks exigem dependências opcionais. As diferenças de nota nos exemplos são explicativas,
não autorização para ajustar parâmetros no GT público.

## Dados e ambiente de cada participante

Siga [DADOS-LOCAIS.md](DADOS-LOCAIS.md). Cada colega obtém seu pacote autorizado; o clone leva
apenas guias em `data/tactical/` e `data/broadcast/`. ATD tactical usa `frame_id - 1`, eventos
usam `round(segundos * 25)` e homografia já usa frames começando em zero. As posições são
normalizadas; converta os eixos para metros antes de comparar distâncias. Confira `valid` nas
homografias. Não há mapeamento pronto entre broadcast e tactical.

`requirements.txt` lista numpy/pandas; `requirements-starter.txt` acrescenta kloppy, matplotlib
e notebook. `requirements-lock.txt` registra o ambiente básico verificado em Python 3.13,
sem incluir todas as ferramentas opcionais.

```bash
python scripts/verificar_dados.py
python scripts/verificar_dados.py --source all
```

`--data-dir` ou `MTAH_DATA_DIR` permitem guardar os dados em outro disco. A conferência de presença
não valida o conteúdo completo ou a sincronização dos vídeos.

## Licença e arquivos auxiliares

[LICENSE](../LICENSE) aplica MIT ao material público, com preservação dos avisos e sem garantia;
não licencia os assets privados. [CITATION.cff](../CITATION.cff) informa como citar o benchmark.
`.gitignore` mantém dados, ZIPs, ambiente e saídas fora do Git; `.ignore` auxilia buscas no mapa
local `graft/`. Nunca force a inclusão dos assets ignorados.

Os demais guias locais são [ORGANIZACAO.md](ORGANIZACAO.md), com as verificações executadas,
e [PESQUISA-INICIAL.md](PESQUISA-INICIAL.md), com referências e hipóteses sem atribuições.
