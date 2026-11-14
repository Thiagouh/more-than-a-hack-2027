# Tolerância independente de bola ausente

O consumidor provisório da frente 4 aceita uma janela de ausência separada dos limiares de aquisição/perda. Ausência de tracking não demonstra bola morta nem troca de posse.

## API e comportamento

`src.team_possession.possession(events, meta, ball_observed=..., missing_ball_grace_frames=5)` recebe uma lista/tupla com um bool por frame. Cada indicação precisa estar disponível causalmente naquele frame; não fornecer retrospectivamente uma observação atrasada como se ela já fosse conhecida. O adaptador ATD oferece disponibilidade no próprio frame.

- Aquisição continua configurável em 3/5 frames; perda é independente.
- O consumidor pode manter o último estado durante os primeiros cinco frames ausentes consecutivos; no sexto emite X e invalida a aquisição ativa.
- Voltar a enxergar a bola não restaura sozinho a aquisição expirada. É necessária nova aquisição com suporte posterior à lacuna.
- Aquisições atrasadas anteriores à lacuna não ressuscitam o estado; aquisições entregues durante uma ausência já expirada são recusadas.
- Descontinuidades reiniciam o contador e o estado. Não há reescrita do passado nem produção automática de D.
- Zero significa X já na primeira observação ausente. Na frequência de 25 Hz, cinco amostras ausentes cobrem 200 ms; o comportamento de fronteira é definido pela contagem acima.

O prazo de cinco é uma hipótese inicial, não um parâmetro validado. Sem `ball_observed`, o comportamento event-only anterior é mantido para compatibilidade; a nova política é opt-in e usa versão própria. `missing_ball_timeouts` conta episódios que excederam o prazo, inclusive se o estado já era X.

## Execução congelada com referência pública

Em 14/11/2026 foi avaliado um recorte fixo de 120 segundos do CSV público usado pelo notebook 03, frames tactical 11185–14184. Os eventos das versões 3/5, com perda em 3 e velocidade relativa máxima 5 m/s, foram reutilizados sem recalibrar o detector. A janela, parâmetros e hashes foram gravados antes de abrir a referência.

O vídeo demonstrativo original mencionado na conversa não foi identificado. O notebook 02 gera uma animação de tracking de dez segundos; o 03 avalia o CSV de referência pública, sem um vídeo incorporado de dois minutos. Foi gerada uma animação nova do tracking real no recorte escolhido, com referência e duas previsões lado a lado. Ela não é filmagem nem verificação dos uniformes dos goleiros.

Resultados e análise permanecem em `outputs/public-2min-missing-ball-v1/ANALISE.md` e `report.html`, com vídeo de exatamente 120 s / 3.000 frames, previsões, comparação por frame e métricas oficiais. Os scores são uma conferência de integração nessa janela pequena, não critério para selecionar parâmetros ou estimativa de ranking na partida inteira.

## Reprodução

A partir da raiz, usar destinos novos:

```sh
.venv/bin/python -m scripts.evaluate_missing_ball \
  --run outputs/persistence-real-3-vs-5-v1 \
  --tracking data/tactical/BAR-ATH_tactical_atd.txt \
  --ground-truth evaluation/gt/BAR-ATH_possession_gt_public.csv \
  --grace 5 --out outputs/public-2min-repeticao
```

A referência só é aberta depois de produzir e salvar as previsões fixas. O avaliador oficial não é modificado. O script verifica hashes dos inputs da execução congelada e a estrutura dos CSVs.

Para o vídeo, instalar a dependência opcional `requirements-visual.txt` no ambiente local e disponibilizar ffmpeg:

```sh
.venv/bin/python -m scripts.render_public_comparison \
  --run outputs/public-2min-repeticao --data-dir data \
  --ground-truth evaluation/gt/BAR-ATH_possession_gt_public.csv
```

A renderização usa 25 fps e mantém frames tactical originais. Cores dos pontos são convenções de time, não cores de uniforme. Assets, vídeo e derivados continuam em diretórios ignorados pelo Git.

## Verificação

83 testes passaram, incluindo os casos de tolerância curta, expiração na sexta ausência, prazo zero, independência dos 3/5 frames, aquisição antiga atrasada, aquisição entregue durante lacuna expirada e exigência de nova evidência após retorno. O vídeo foi conferido como 960×820, 25 fps, 3.000 frames e 120 segundos. Os três arquivos oficiais de avaliação permanecem preservados.

## Extensão para aquisição de 1, 2 e 4 frames

A aquisição agora aceita `acquisition_persistence_frames=1`: confirma no primeiro frame elegível, ainda exigindo histórico de movimento, qualidade e disponibilidade. `loss_persistence_frames` mantém mínimo 2; o padrão global continua 3. O esquema de evento v2 e seu validador foram ampliados apenas para aquisição.

Os mesmos 20 cenários sintéticos e o mesmo recorte público de 120 segundos foram executados com 1/2/3/4/5, mantendo perda=3, bola ausente=5 e velocidade relativa=5. Os perfis anteriores 3/5 reproduziram exatamente as previsões salvas. Esta rodada processou apenas o prefixo causal 0–14184, suficiente para preservar todo o histórico do recorte, sem reprocessar a partida inteira.

Resultados e nova animação 1/2/4: `outputs/public-2min-profiles-1-2-3-4-5-v1/report.html`. Foram aprovados 88 testes automatizados. Para reproduzir com um destino novo:

```sh
.venv/bin/python -m scripts.compare_acquisition_profiles \
  --data-dir data --baseline outputs/public-2min-missing-ball-v1 \
  --ground-truth evaluation/gt/BAR-ATH_possession_gt_public.csv \
  --out outputs/public-perfis-repeticao
.venv/bin/python -m scripts.report_acquisition_profiles outputs/public-perfis-repeticao
.venv/bin/python -m scripts.render_public_comparison \
  --run outputs/public-perfis-repeticao --data-dir data \
  --ground-truth evaluation/gt/BAR-ATH_possession_gt_public.csv --profiles 1 2 4
```

A execução grava automaticamente `reference-sha256.json` com o hash validado da referência; o renderer usa esse registro para não misturar gabaritos. Nenhum parâmetro deve ser escolhido pela janela pública.
