# Frente 4 — baseline refeito e regras de posse

Estado em **16/11/2026**. O pacote `src/frente4/` transforma os dados das frentes 1 a 3 e os
eventos da frente 5 em um label H/A/D/X por frame e escreve o CSV da submissão.
Nenhum módulo lê o gabarito; a janela pública entra só como conferência de integração.

## Comando único

```sh
.venv/bin/python -m src.frente4 run --out outputs/submissions/frente4.csv
.venv/bin/python evaluation/validate_submission.py outputs/submissions/frente4.csv
```

Opções: `--data-dir` (ou `MTAH_DATA_DIR`), `--config configs/frente4_v1.json`,
`--events outputs/frente5/<execução>/events.jsonl` (eventos v2 da frente 5, opcional),
`--cache-dir` (padrão `outputs/cache/`). Ao lado do CSV sai um `*.diagnostico.json` com contagem
de posses, duração média, motivos de D, fração de bola observada e os parâmetros usados.

Outros comandos:

| Comando | O que faz |
|---|---|
| `python -m src.frente4 baseline --out outputs/submissions/baselines` | Refaz os quatro baselines da organização com código nosso (vetorizado sobre o cache) e diz se cada CSV saiu **idêntico** ao de `evaluation/results/` |
| `python -m src.frente4 from-csv <csv> --out <csv>` | Roda só a máquina de estados sobre um CSV de candidatos (H/A = candidato, D/X = ninguém). Serve sem os dados da partida |
| `python -m src.frente4 score <csv> ...` | Nota na janela pública com o scorer oficial (conferência, não ajuste) |

## Arquitetura

```
ATD txt+xml ──atd.py──> cache .npz ──features.py──> features por frame ──┐
Smart Tagging ──────────deadball.py──> dead[], hint_team[] ─────────────┤
eventos v2 (frente 5) ──pipeline.events_to_team_array──> event_team[] ──┤
                                                                        └──fsm.py──> códigos ──> CSV
```

| Módulo | Responsabilidade | Dono natural |
|---|---|---|
| `atd.py` | Lê o ATD uma vez (pandas), converte para metros, guarda `.npz` em `outputs/cache/` com chave por tamanho/mtime dos arquivos. Devolve `players_xy [n,46,2]`, `ball_xy [n,2]`, `team_of_slot [46]` | Frente 1 (entrega provisória aqui; trocar mantendo as chaves) |
| `features.py` | Distância do mais próximo de cada time, margem, candidato dentro do raio, jogadores perto da bola, velocidade da bola e média dos jogadores. Só mede | Frente 1 |
| `deadball.py` | Evidência de D por frame e pista de time para reposições | Frente 3 |
| `fsm.py` | Máquina de estados: candidato por frame -> H/A/D/X com memória | Frente 4 |
| `baseline.py` | Baseline da organização refeito; `smooth()` idêntico ao do starter | Frente 4 |
| `pipeline.py` / `__main__.py` | Cola e CLI | Frente 4 |

## Candidato por frame

`cand[f]` é o time do jogador mais próximo da bola, **se** ele está a até `control_radius_m`
(3 m) **e** a margem para o adversário mais próximo é maior que `contested_margin_m` (0,5 m)
**e** a bola corre a menos de `max_control_speed_mps` (10 m/s). Em qualquer outro caso o frame
não tem candidato: bola ausente, bola solta, bola disputada e bola em voo não trocam posse.

Quando a frente 5 entrega `events.jsonl`, cada `possible_control_acquisition` com time vira
candidato forte por `event_hold_frames` (12) frames, sobrepondo o candidato geométrico.

## Máquina de estados (`fsm.py`)

| Regra | Parâmetro | Padrão |
|---|---|---|
| Tempo mínimo para trocar: o adversário precisa ser candidato em N frames | `min_switch_frames` | 12 (0,48 s) |
| Buracos sem candidato tolerados dentro dessa sequência | `max_gap_frames` | 5 |
| Troca retroativa: confirmada a troca, o novo time recebe desde o **primeiro** frame da sequência | — | sempre |
| Bola some ou ninguém controla: manter o último estado | — | sempre |
| Depois de D ou do início do tempo: o primeiro time que controla por N frames recebe retroativamente desde o fim do D | `restart_confirm_frames` | 3 |
| Pista de time (quem cobra) usada só enquanto se espera o primeiro controlador | `hint_team` | Smart Tagging |
| Evidência de D mais curta que N frames é ignorada | `min_dead_frames` | 5 |
| Fora dos blocos avaliados (antes do apito, intervalo, depois do fim) | — | X |

Por que assim: o ranking é `spellF1`, que casa blocos inteiros de posse por IoU temporal
(docs/EVALUATION.md). Trocar a cada frame ruidoso fragmenta posses e zera o casamento; atrasar
uma troca real em 1 s custa ~35 pontos. Por isso a confirmação é curta e retroativa.
A frente anterior (`src/team_possession`) é causal e emite X sem bola; aqui tudo é offline e
olhar para frente é legítimo, porque o CSV é gerado de uma vez.

## Regras de D (`deadball.py`)

| Fonte | Regra | Parâmetros |
|---|---|---|
| Antes do pontapé inicial | Do primeiro frame avaliado até o kickoff publicado. No 2º tempo são os **2.934 frames 110808–113741**; no 1º o kickoff coincide com o primeiro frame avaliado | `KICKOFFS = {1: 11185, 2: 113742}` |
| Bola fora | Bola observada além da linha por mais de `out_margin_m` em `out_min_frames` consecutivos abre D | 0,5 m, 3 frames |
| Set pieces do Smart Tagging | Escanteio, tiro de meta, falta, lateral, kick off após gol: se ainda não está D no início da marcação, abre D ali. `Team` vira pista de time (lateral vem como N/A e não dá pista) | `use_set_piece_tags` |
| Bola parada | Bola observada a menos de `still_speed_mps` por `still_frames` abre D retroativo ao início da parada | 0,5 m/s, 100 frames (4 s), `use_stillness` |
| Reposição (fecha o D) | Primeiro frame com bola dentro do campo e a `restart_speed_mps` ou mais, depois de `min_dead_after_trigger_frames` | 3 m/s, 25 frames |
| Teto | Um D nunca passa de `max_dead_frames` | 2250 (90 s) |

**Bola ausente não é D.** Medido no `baseline_raw.csv` da organização (que codifica a
visibilidade da bola): são apenas 150 sumiços nos frames avaliados, cobrindo 40,7% deles, com
mediana de 268 frames (10,7 s). Na janela pública, dos quatro sumiços de mais de 2 s, dois caem
em D do gabarito e dois caem com a bola em jogo (posse A). Logo o sumiço longo não discrimina;
a máquina de estados mantém o último time e a frente 2 (estimativa da bola pelo conjunto)
é quem melhora esses trechos. Já dois dos quatro D do gabarito na janela têm a bola 100% visível:
são os casos das regras de bola fora/parada/set piece.

## O que já foi conferido

- 20 testes sintéticos em `tests/test_frente4.py` (máquina de estados, regras de D, features,
  baseline). Rodam no CI com `python -m unittest discover -s tests`.
- `smooth()` reproduz **byte a byte** `baseline_smooth25/50/125.csv` a partir do
  `baseline_raw.csv` oficial; e é idêntico ao `starter.baseline.smooth` em sequências aleatórias.
- A máquina de estados sobre o `baseline_raw.csv` (candidato sem raio, sem margem, sem
  velocidade) reduz 131 -> 66 posses na janela pública (gabarito: 17). Isso mostra o limite do
  sinal "mais próximo" puro: dentro de uma posse H de 41 s ele tem 11 trechos A, o maior com
  3,5 s. Os filtros de raio, margem e velocidade existem para isso, mas só podem ser medidos com
  o ATD local.

## Pendente (precisa dos dados locais)

1. `python -m src.frente4 baseline` tem que imprimir `idêntico ao oficial: True` nas quatro
   variantes; isso valida cache, metros e mapa de times. Nota pública esperada do smooth50: 0,3256.
2. Rodar `run`, validar o CSV e olhar `diagnostico.json`: número de posses da partida inteira
   (ordem de grandeza: algumas centenas), duração média e motivos de D.
3. Integrar `events.jsonl` da frente 5 e comparar com/sem eventos usando
   `evaluation/comparar_csv.py` (diferença, não nota).
4. Validar limiares em outros jogos (frente 5), nunca na janela pública.

## Limitações conhecidas

- Sem estimativa da bola nos sumiços longos (frente 2), a posse fica congelada por até dezenas
  de segundos; uma troca que acontece com a bola invisível só é percebida quando ela volta.
- A regra de bola parada pode marcar D quando o goleiro segura a bola; a de bola fora depende
  da precisão do tracking perto das linhas. Ambas têm chave para desligar.
- Set pieces do Smart Tagging são janelas de ~5 s em torno da reposição; se a marcação começa
  depois da reposição real, o D entra atrasado e dura pelo menos 1 s.
