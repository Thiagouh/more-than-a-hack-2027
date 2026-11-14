# Avaliação da equipe — frente 5

## Primeiro passo

Executar da raiz do repositório. Ambiente local:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
```

Os testes usam somente sequências sintéticas. Não leem tracking, vídeo ou gabarito da partida.
O avaliador oficial permanece intacto; não implementamos uma métrica paralela.

## O que já foi confirmado no material local

- Entrega: 199.319 frames, numerados de 0 a 199318 (`docs/SUBMISSION.md`).
- Ranking: `spell_f1_mean`, média do F1 de blocos H/A nos limiares de IoU temporal 0,3/0,4/0,5/0,6/0,7. MacroF1 é a visão por frame, não a métrica principal.
- D não é posse, mas interrompe um bloco H/A. H-D-H produz duas posses H.
- X previsto conta como erro no macroF1 em frame avaliado. Nas métricas temporais, é preenchido com o último label conhecido dentro de cada intervalo avaliado definido pelo gabarito; no início, usa o primeiro label conhecido. Um intervalo inteiramente X permanece X.
- O preenchimento pode repetir D, não apenas H/A. Nunca atravessa lacunas X do gabarito.
- Sem gabarito, a contagem bruta trata X como interrupção. Não equivale à contagem do avaliador.
- A janela pública tem apenas 17 posses. Usá-la para conferir alinhamento e funcionamento, não para selecionar parâmetros ou versões (`docs/EVALUATION.md`). Bootstrap não elimina sobreajuste.

## Divergências com o resumo da reunião

O `docs/RULES.md` local permite pequenas anotações manuais para treino/validação, mas proíbe produzir ou corrigir manualmente a submissão. Também menciona métodos com LLMs e APIs, sujeitos à inspeção e documentação; os termos de inscrição governam a licença dos dados. Isso difere do resumo recebido. Até confirmar a regra aplicável com a equipe/organização, manter o fluxo conservador: dados da partida locais, nenhuma anotação manual e exemplos sintéticos neste trabalho.

O resumo cita 1.999 frames; o contrato local exige 199.319. O repositório tem nome de edição 2027, enquanto `SUBMISSION.md` declara prazo em 17/11/2026. Confirmar a edição e a versão aplicável do regulamento antes da entrega.

## Comparar duas previsões sem gabarito

```sh
.venv/bin/python evaluation/comparar_csv.py caminho/a.csv caminho/b.csv --rows 199319 --out outputs/comparacao
```

Valida com as funções oficiais: labels, células vazias, sequência de frames, duplicatas e total de linhas. Sem `--rows`, permite sequências sintéticas de qualquer tamanho, mas exige cobertura igual, contígua e iniciada em zero. Não certifica então o tamanho oficial.

Gera `comparacao.json` (distribuição de labels, posses brutas, duração média, cruzamento de labels e diferenças) e `divergencias.csv` (intervalos contíguos diferentes, limites inclusivos). `--fps` controla apenas a conversão de duração, padrão 25.

Diferença não significa melhora. D e X não são intercambiáveis. Não usar número de posses ou percentual de D como objetivo isolado.

## Fluxo oficial, quando necessário

Validar uma submissão sem consultar o gabarito:

```sh
.venv/bin/python evaluation/validate_submission.py caminho/submissao.csv
```

Conferir a integração na janela pública, sem varrer parâmetros:

```sh
.venv/bin/python evaluation/run_evaluation.py --gt public --results evaluation/results --out outputs/avaliacao-publica --bootstrap 1000
```

Esse comando avalia todos os CSVs da pasta indicada. O avaliador já gera tabela, relatórios JSON e transições CSV. Não precisa ser copiado ou reescrito. Ainda não foi executado contra dados reais nesta preparação.

## Próximas entregas

1. Visualização da linha do tempo H/A/D/X, preservando os frames originais.
2. Registro de experimentos: hipótese anterior à execução, versão do código, parâmetros, origem dos dados, comando, resultados e limitações. Dados reais e saídas ficam fora do Git público.
3. Protocolo para dados externos: disponibilidade, compatibilidade, definição de controle, referência independente e separação entre ajuste e avaliação.
4. Detector inicial de eventos; ajustar parâmetros em referências permitidas e independentes, mantendo a janela pública como verificação de integração.

## Linha do tempo visual

A implementação usa SVG, sem dependências Python adicionais. Para PNG requer `rsvg-convert` (librsvg), encontrado nesta máquina. Em outras máquinas, exportar SVG funciona sem esse conversor; Matplotlib não é necessário.

Gerar os dois CSVs sintéticos de 600 frames (24 segundos) e as imagens SVG/PNG:

```sh
.venv/bin/python scripts/demo_timeline.py
```

A demonstração compara um cenário coerente com uma variante que introduz fragmentação (frames 60–69), X (120–139), atraso na transição (200–224) e D adicional (490–499). São exemplos construídos por código, não dados reais nem medição de desempenho.

Desenhar arquivos próprios:

```sh
.venv/bin/python evaluation/desenhar_posses.py outputs/demo_timeline/cenario_coerente.csv outputs/demo_timeline/cenario_com_falhas.csv --inicio 100 --fim 250 --out outputs/demo_timeline/recorte.png
```

O intervalo é inclusivo, todas as faixas usam a mesma escala e o eixo mantém o tempo desde o frame zero. Cada barra cobre até o fim do último frame, isto é, `(fim + 1) / fps`. H azul, A laranja, D cinza, X amarelo. X permanece visível. Nomes longos são abreviados; o SVG mantém o nome completo e os limites das barras em dicas ao passar o cursor. `--rows 199319` pode exigir o tamanho oficial; sem isso, aceita exemplos menores, sempre com cobertura igual e iniciada em zero.

## Implementação da frente 5 — 03/11/2026

O módulo e seus comandos estão em [FRENTE5-EVENTOS.md](FRENTE5-EVENTOS.md). A pesquisa e recomendação externa estão em [DADOS-EXTERNOS-FRENTE5.md](DADOS-EXTERNOS-FRENTE5.md). A frente 5 entrega eventos candidatos; a decisão final permanece na frente 4.

Use `.venv/bin/python` **dentro da raiz do repositório**, não o ambiente homônimo na pasta superior. Os testes e a demo continuam sintéticos. Não houve avaliação da partida nem consulta de gabarito nesta implementação.

## Correção após revisão — 04/11/2026

O contrato de eventos foi atualizado para v2, com predecessores de perdas e entrega causal. Ver [guia atualizado](FRENTE5-EVENTOS.md) e [matriz da revisão](RESOLUCAO-REVISAO-FRENTE5.md). Entradas continuam v1, com validação mais rigorosa. Exportações antigas não certificam o contrato novo: regenerar, sem fabricar dependências. O avaliador oficial continua preservado.
