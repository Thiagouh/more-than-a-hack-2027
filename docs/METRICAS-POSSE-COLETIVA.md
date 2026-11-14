# Métricas propostas para posse coletiva e passes

Data: 14/11/2026. **Status: implementada como experimento v1.** Métricas em `src/team_possession/collective.py` e `kinematics.py`; avaliações em `src/team_possession/evaluation.py`. Os valores físicos abaixo são hipóteses explícitas, não calibração esportiva. Os resultados locais e a animação estão vinculados ao final.

## 1. Objetivo e decisões escolhidas

Preservar a hipótese de posse de um time enquanto a bola viaja em um passe plausível, distinguindo isso de controle individual, disputa, recuperação adversária e falta de informação.

Escolhas para a primeira implementação:

1. Usar o último controle confirmado como origem; atualizá-lo com evidência corrente, sem depender da identidade pessoal.
2. Prever a trajetória apenas com observações já disponíveis e medir o erro dessa previsão.
3. Estimar a primeira oportunidade de encontro com a bola para cada time, ao longo do percurso, não só no destino.
4. Separar oportunidade adversária de interceptação de controle adversário efetivamente observado.
5. Manter métricas em unidades físicas. **Não criar uma soma de pesos nem chamar um score heurístico de probabilidade.**
6. Manter aquisição em 3 frames como referência experimental; perda em 3 e tolerância independente de bola ausente em 5. Esses valores não são declarados ótimos.

A saída continua H/A/D/X, mas o estado interno proposto distingue `controle`, `passe`, `disputa` e `incerteza`. D precisa de um mecanismo de bola morta próprio; nenhuma métrica abaixo o produz sozinha.

## 2. Convenções e causalidade

- Posições em metros, velocidades em m/s, tempos em segundos, ângulos em graus. H/A vêm do time do rastro; goleiros participam do respectivo time.
- `t` é o instante estimado da observação; uma medida só pode ser usada no seu `available_at_frame`. Nunca usar o recebedor que de fato recebeu no futuro para justificar uma decisão anterior.
- Origem, dados, previsões e eventos não atravessam descontinuidades conhecidas. Velocidade individual usa apenas o mesmo rastro; a posse coletiva pode continuar entre rastros diferentes.
- Valores indisponíveis são `null` com motivo, nunca zero. `0 s` significa encontro possível agora; não significa ausência de estimativa.
- Bola sem observação não ganha uma trajetória inventada para continuar confirmando posse. O prazo de ausência é independente do tempo de voo previsto.
- `observed` é uma saída do tracking, não uma posição certificada. Flags, saltos e duplicatas continuam sendo considerados.

### Estimativa de movimento escolhida

Para as novas métricas preditivas, ajustar uma reta por mínimos quadrados em cada coordenada às **últimas até 5 observações consecutivas válidas**, exigindo no mínimo 3. A inclinação estima a velocidade; o ajuste permite medir ruído. Em 25 Hz, cinco observações cobrem 160 ms entre a primeira e a última.

A escolha é um ponto de partida causal que reduz a dependência de uma única diferença entre frames. Pode atrasar a percepção de uma mudança brusca; por isso manter também a diferença instantânea e o erro de previsão. Não substituir silenciosamente a derivada atual do detector: publicar ambas com nomes e suportes distintos.

Ausência, salto ou troca de rastro interrompe o ajuste correspondente; não preencher a janela com pontos interpolados para confirmar controle. Com menos de três amostras, as novas previsões de movimento ficam indisponíveis. O histórico e a disponibilidade efetivamente usados devem acompanhar a medida.

## 3. Origem do passe

| ID / campo | Definição | Unidade / interpretação |
|---|---|---|
| O1 `origin_team` | Time da aquisição confirmada que sustenta a hipótese corrente | H/A/null; contexto, não prova de posse atual |
| O2 `last_supported_control_age_s` | `t − t_ultima_evidencia_valida_de_controle` desse time | s; envelhecimento da evidência |
| O3 `origin_support_fraction` | Frames com evidência de controle elegível do time / total de frames da janela recente de 5 | 0–1, fração observada; não probabilidade |
| O4 `release_interval` | Intervalo entre o último frame com controle sustentado e o primeiro frame de separação válida que abre a hipótese de passe | Dois frames e largura em ms; incerteza sobre a saída da bola |

**Detalhe necessário:** atualizar O2 com o último frame de evidência elegível durante controle ativo, não apenas com a data do primeiro evento de aquisição. Caso contrário, uma condução longa pareceria uma origem cada vez mais antiga. O detector agora expõe elegibilidade de controle por time e o suporte correspondente nas features.

O3 sempre acompanha a fração de observações válidas da mesma janela. Um frame desconhecido não conta como controle, mas sua ausência deve aparecer explicitamente no diagnóstico. O4 não é automaticamente o instante exato de um chute: o afastamento de 2 m pode ocorrer depois do contato.

## 4. Trajetória da bola

Primeiro modelo escolhido: **velocidade constante em XY**, reestimada a cada observação válida:

`b_prevista(t + τ) = b(t) + v_b(t) × τ`

É uma hipótese local, não um modelo físico completo de voo. O horizonte deve ser curto e configurável; um candidato além dele é **censurado pelo horizonte**, não declarado impossível.

| ID / campo | Definição | Unidade / interpretação |
|---|---|---|
| T1 `ball_speed_fitted_mps` | Norma da velocidade ajustada da bola | m/s; avanço da bola |
| T2 `ball_heading_deg` | Direção do vetor de velocidade | graus; indisponível quando o movimento é pequeno demais para direção confiável |
| T3 `trajectory_error_m` | Distância entre a observação atual e a previsão para este frame feita anteriormente | m; surpresa da trajetória, sem reajustar a previsão com o próprio ponto avaliado |
| T4 `trajectory_error_rms_m` | Raiz da média dos quadrados dos últimos até 5 erros T3 válidos | m; estabilidade recente da previsão |
| T5 `speed_change_mps` | Diferença assinada entre velocidades escalares consecutivas válidas | m/s; aceleração/desaceleração aparente, não aceleração em m/s² |
| T6 `heading_change_deg` | Menor diferença angular absoluta entre direções consecutivas válidas | 0–180°; possível desvio, não comprovação de toque |

Não confundir T4 com o erro do ajuste sobre seus próprios pontos: a medida principal avalia previsões anteriores. Não interpretar T4 como um intervalo de confiança de 95% sem calibrar sua cobertura.

Um T3/T6 elevado pode decorrer de toque, quique, bola aérea projetada no chão ou falha do tracker. Para atribuir intervenção, exigir evidência espacial e temporal adicional.

## 5. Recebedores e adversários: encontro com a bola

### Modelo de chegada

Para cada jogador válido `i` e ponto futuro `q`, estimar `T_i(q)`: tempo para alcançar a vizinhança controlável de q, considerando posição, velocidade atual, reação, aceleração e velocidade máxima do modelo. É tempo de **oportunidade**, não garantia de execução de uma ação.

O modelo deverá explicitar `reaction_time_s`, `acceleration_mps2`, `max_speed_mps` e `control_radius_m`. Não fixar números físicos novos neste documento como se fossem medidos. A primeira implementação deve registrá-los e submetê-los à calibração/sensibilidade antes de usar o resultado como regra decisiva. O raio de controle atual de 1,5 m pode ser o ponto de partida, com seu status heurístico preservado.

Considerar o movimento durante a reação e a direção do deslocamento; não transformar um jogador correndo para longe em um jogador parado já orientado para q. O modelo aproximado ainda terá limitações de giro e deslocamento lateral.

Avaliar pontos da trajetória nos instantes `τ = Δt, 2Δt, ... H`, com `Δt = 1/fps` e horizonte H configurável. Para cada jogador:

`τ_i = primeiro τ em que T_i(b_prevista(t + τ)) ≤ τ`

Assim, a bola precisa chegar ao ponto ao mesmo tempo em que o jogador consegue alcançá-lo. Testar apenas a distância ao destino não basta: um adversário pode interceptar antes.

### Métricas escolhidas

| ID / campo | Definição | Unidade / interpretação |
|---|---|---|
| R1 `origin_team_meeting_time_s` | Menor `τ_i` entre companheiros da origem | s; primeira oportunidade prevista do time |
| R2 `receiver_candidates_count` | Número de rastros válidos do time com encontro viável dentro de H | Contagem; não somar como votos de confiança |
| R3 `receiver_approach_distance_m` | Menor distância prevista bola–jogador no horizonte, usando o movimento corrente, para os candidatos | m; compatibilidade geométrica, distinta da capacidade potencial de correr até a bola |
| R4 `receiver_relative_speed_mps` | Norma da diferença entre velocidades observadas de bola e candidato durante a aproximação | m/s; evidência de recepção quando há observação atual próxima |
| A1 `opponent_meeting_time_s` | Menor `τ_i` entre adversários válidos | s; primeira oportunidade adversária no percurso |
| A2 `meeting_time_advantage_s` | `opponent_meeting_time_s − origin_team_meeting_time_s` | s; positivo favorece a origem; negativo favorece o adversário |
| A3 `opponent_exposure_fraction` | Fração dos instantes previstos em que algum adversário pode alcançar a bola até aquele instante | 0–1, exposição geométrica ao longo da previsão; não probabilidade de interceptação |
| A4 `opponent_motion_change_evidence` | T3/T5/T6 associados a proximidade adversária e evento de toque, com suporte temporal explícito | Registro de evidências; desvio possível, não troca automática |
| A5 `opponent_control_streak_frames` | Evidências consecutivas elegíveis de controle adversário | Frames e duração; confirmação de aquisição permanece em 3 como referência |

R3 usa projeção do movimento corrente para checar compatibilidade; R1/A1 usam a capacidade potencial do modelo de chegada. Esses conceitos não podem ser apresentados como a mesma previsão.

Se um time não tem candidato no horizonte, seu tempo fica `null` com motivo `no_meeting_within_horizon`. Se faltam dados, usar motivo `insufficient_coverage` ou `motion_unavailable`. Não substituir por infinito e calcular uma vantagem artificialmente enorme. A2 fica indisponível nesses casos, acompanhada dos dois motivos.

Duas opções igualmente boas do mesmo time não prejudicam a posse coletiva. Os rastros candidatos são explicações locais, não identidades pessoais persistentes. Duplicatas não contam como recebedores adicionais.

## 6. Qualidade e incerteza obrigatórias

| ID / campo | Definição | Uso |
|---|---|---|
| Q1 `ball_missing_run_frames` | Número de frames consecutivos sem bola observada | Aplicar tolerância independente; não inferir D |
| Q2 `observed_players_by_team` | Número de jogadores com posição observada e válida por time | Diagnosticar cobertura; não assumir sempre 11 jogadores visíveis |
| Q3 `valid_motion_fraction_by_team` | Jogadores observados com movimento válido / jogadores observados, por time | Identificar vantagem de interceptação calculada com cobertura desigual |
| Q4 `evidence_age_s` | Idade das observações que sustentam trajetória e candidatos | Não tratar evidência antiga como atual |
| Q5 `forecast_horizon_s` e `censored_candidates` | Horizonte utilizado e candidatos sem encontro dentro dele | Distinguir censura de impossibilidade |
| Q6 `quality_flags`, suporte e disponibilidade | Ausências, saltos, duplicatas, pontos estimados, frames usados e instante em que a medida está disponível | Auditoria e bloqueio de confirmação |

Q2/Q3 descrevem os objetos que o tracker vê, não a cobertura real de todos os jogadores em campo. A ausência de um defensor pode tornar um passe artificialmente seguro. Portanto, nenhuma vantagem favorável ao passe basta sozinha para confirmar posse.

Sem altura da bola, a oportunidade de interceptação em XY pode superestimar a capacidade de alcançar uma bola aérea. Marcar essa limitação; não declarar um passe rasteiro ou interceptável apenas por sua projeção no campo.

## 7. Uso pretendido na decisão

A política foi implementada como experimento opt-in, com os parâmetros explícitos da seção 11. Não é um modelo calibrado nem substitui a avaliação independente.

| Situação | Evidência relevante | Conduta proposta |
|---|---|---|
| Saiu do raio de controle, mas há passe plausível | O1/O2 válidos, trajetória coerente e R1 disponível, com qualidade suficiente | Abrir estado `passe` mantendo o time de origem |
| Adversário pode chegar antes | A1/A2/A3 | Abrir hipótese de disputa/ameaça; não confirmar troca apenas pela previsão |
| Houve desvio perto de adversário | A4 | Reestimar trajetória e recebedores; manter intervenção distinta de controle |
| Adversário estabelece controle | A5 e aquisição validada | Trocar o time na disponibilidade real da confirmação |
| Recebedor do mesmo time controla | R4 e aquisição/evidência válida | Voltar a `controle` sem criar troca de time |
| Trajetória perdeu sustentação ou bola ausente além da tolerância | T3/T4/Q1/Q4/Q6 | X quando a hipótese já não é sustentada; não conservar time indefinidamente |

O envelhecimento O2 deve ser interpretado junto do tempo de voo: um passe longo observado pode demorar mais que um passe curto. Não usar o mesmo timeout fixo para ausência, perda de controle e toda duração de passe. Ao mesmo tempo, o horizonte H não autoriza manter um passe indefinidamente: registrar idade da hipótese, candidatos e erro a cada atualização.

## 8. Métricas para avaliar se isso melhora o modelo

Estas exigem uma referência independente do que se quer medir. O CSV público H/A/D informa posse por frame, mas **não identifica sozinho passe, recebedor, toque ou instante de controle**. Não fabricar essas anotações a partir de nossa própria previsão.

| Métrica | Definição e denominador | O que revela |
|---|---|---|
| Retenção correta durante passe | Frames previstos com o time correto / frames de passes na referência | Continuidade coletiva; X conta como ausência de resposta |
| Passes preservados integralmente | Passes sem X nem time errado em todo o intervalo / passes de referência | Critério estrito por ação, acompanhado da duração dos passes |
| Falsas trocas por 100 passes | Passes com ao menos uma indicação indevida do adversário / passes de referência × 100 | Trocas artificiais; X não é contado como troca |
| Abstenção durante passe | Frames X / frames de passes na referência | Se apenas escondemos o erro com X |
| Precisão e recall de recuperações | Pareamento um a um entre recuperações previstas e de referência, mesma equipe e tolerância temporal fixada antes da execução | Interceptações recuperadas e trocas espúrias |
| Erro do instante de recuperação | Erro assinado, MAE e percentis dos instantes estimados pareados | Localização temporal do evento, separada da entrega |
| Atraso de confirmação e entrega | Instantes de confirmação/disponibilidade menos instante da referência, com distribuição e contagem de antecipações | Latência real e decisões precoces; não descartar erros negativos |
| Erro de previsão da bola | Mediana, RMSE e P90 de T3, separados por horizonte | Se a trajetória realmente antecipa o movimento |
| Erro de encontro previsto | Erro assinado, MAE e P90 contra oportunidades/recepções com referência apropriada | Qualidade dos tempos previstos, sem chamar uma chegada possível de recepção garantida |
| Qualidade final de posse | F1 temporal oficial, Macro F1, matriz de confusão H/A/D/X e cobertura de respostas | Resultado global; mostrar abstenção junto do score temporal |

Reportar por time, goleiro/jogador de linha quando houver papel conhecido, passes curtos/longos, disputa, cobertura e bola aérea quando a referência permitir. Informar contagens e denominadores, não só médias agregadas. Não converter a duração de posse em alvo de otimização.

As primeiras provas devem ser sintéticas com expectativas pré-definidas: passe curto, passe longo, duas opções do mesmo time, interceptação no meio do corredor, desvio sem controle, oclusão curta/longa e defensor ausente do tracking. Para estimar precisão esportiva, usar dados de outros jogos e referências permitidas separadas entre calibração e avaliação. A janela pública já consultada não é um teste independente para escolher o melhor parâmetro.

## 9. Entrega observável e prioridades

**Prioridade 1:** O1/O2/O4, T1/T3/T4, Q1/Q2/Q3/Q6 e distinção entre controle e posse. Essas medidas explicam por que o modelo abre e encerra uma hipótese de passe.

**Prioridade 2:** R1/R2, A1/A2 e A5, com parâmetros físicos e horizonte explicitados. São as medidas centrais para sustentar recebedores e ameaças de interceptação.

**Prioridade 3:** R3/R4, T5/T6, A3/A4, calibração dos erros e avaliação estratificada. Aumentam o diagnóstico e a discriminação de intervenções.

Publicar por frame/episódio: versão da métrica, valores, unidade, validade, motivo de indisponibilidade, time de origem, estado, candidatos, suporte temporal, disponibilidade, parâmetros e motivo da decisão. Não alterar o contrato público de eventos sem decidir como essas novas features serão versionadas.

Na simulação visual: mostrar bola e jogadores, origem, corredor previsto, candidatos dos dois times, tempos R1/A1 e vantagem A2; exibir trajetória incerta ou indisponível como tal. Manter três informações separadas: controle observado, hipótese de posse e gabarito. As cores dos times no diagrama não representam uniformes verificados.

## 10. Referências e limites da escolha

- William Spearman, **Beyond Expected Goals**, 2018: modelo de oportunidades com tempos de interceptação e movimento. Referência conceitual para R1/A1, não origem de limiares calibrados neste projeto: https://static.hudl.com/craft/downloads/SSAC2018_Beyond_Expected_Goals.pdf?mtime=20180327111214
- Javier Fernández e Luke Bornn, **Wide Open Spaces**, 2018: controle espacial com informação de movimento, posição e distância; controle de espaço não equivale automaticamente a posse confirmada: https://www.sloansportsconference.com/research-papers/wide-open-spaces-a-statistical-technique-for-measuring-space-creation-in-professional-soccer

Continuam pendentes de calibração independente o horizonte, os parâmetros físicos, os limites de erro, a cobertura e o prazo de sustentação. A primeira implementação usa as hipóteses explícitas abaixo, verificadas em cenários sintéticos e sem escolha pelo score público.


## 11. Implementação experimental v1

### Parâmetros usados

| Parâmetro | Hipótese inicial | Motivo operacional, não comprovação empírica |
|---|---:|---|
| Persistência de aquisição / perda | 3 / 3 frames | Manter a referência anterior e isolar a política coletiva |
| Ausência tolerada | 5 frames | Separar observação ausente de troca de posse; X na sexta ausência |
| Janela de ajuste | até 5 pontos; mínimo 3 | Derivada causal suavizada, mantendo também a instantânea |
| Horizonte de trajetória | 2 s, reestimado a cada frame | Incluir recebedores de passes mais longos sem usar o futuro observado |
| Reação | 0,2 s | Hipótese explícita do modelo de chegada |
| Aceleração / velocidade máxima | 3 m/s² / 8 m/s | Hipóteses explícitas; movimentos acima do limite ficam fora desse modelo de chegada |
| Raio de controle | 1,5 m | Reusar a geometria heurística existente |
| Erro local de trajetória máximo | 0,75 m | Limite experimental de meia unidade do raio de controle; não equivale a confiança de longo prazo |
| Cobertura mínima para passe | 2 rastros por time e 50% com movimento válido | Evitar chamar um passe de seguro com observações insuficientes; não certifica cobertura dos jogadores reais |
| Prazo sem nova evidência de controle | 4 s | Limitar a sustentação de uma origem antiga, separadamente da ausência de bola |
| Margem temporal de disputa | 0,12 s | Três intervalos a 25 Hz como tolerância experimental; previsão não troca posse |
| Movimento mínimo de passe / direção | 2 m/s / 2 m/s | Reusar a escala mínima de movimento existente; o teste de movimento considera a derivada instantânea ou ajustada |

O limite de erro local usa T3 e T4, ambos retrospectivos a previsões já feitas. Ele **não garante** que uma extrapolação de dois segundos seja precisa. Há diagnóstico posterior por horizonte em `forecast-diagnostics.json`.

### Interfaces, versões e disponibilidade

```python
from src.team_possession.collective import CollectiveModel

model = CollectiveModel(manifest, detector_config={
    'persistence_frames': 3,
    'acquisition_persistence_frames': 3,
    'loss_persistence_frames': 3,
})
for envelope in frames:
    state, feature, new_events, measurement = model.feed(envelope)
quality = model.finish()
```

- `state` contém label, estado, motivo, métricas, validade/motivos de indisponibilidade por campo, suporte, origem, candidatos e previsão disponíveis à decisão.
- `measurement` é o registro estimado, com suporte e disponibilidade próprios. O contrato de métricas é `collective-metrics-v1`; o de política é `collective-pass-v1-experimental`. O catálogo de unidades acompanha a execução.
- A frente 5 acrescenta `control_eligible_team`, `control_evidence_support` e `control_evidence_version` às features. Os eventos públicos continuam v2, sem novos campos obrigatórios ou alteração de suas regras de aquisição.
- O envelope de suporte é conservador: inclui as dependências das métricas e do contexto de origem. Métricas atrasadas não sobrescrevem registros mais recentes.
- Coordenadas ainda indisponíveis no frame são mascaradas como ausência operacional, com flag `input_not_yet_available`, antes de calcular saltos, duplicatas ou movimento. Não podem afetar nem indiretamente uma decisão anterior à disponibilidade. A máscara é auditada; pontos históricos atrasados não são reintroduzidos retrospectivamente na trajetória. Essa política conservadora pode perder evidências úteis.
- A confirmação de recuperação adversária continua vindo de eventos de aquisição válidos, não do tempo de interceptação previsto.

### Avaliações e limites de referência

`evaluate_collective` aceita referências independentes opcionais de posse, passes, recuperações e encontros. Ausência de referência retorna `available=false` e um motivo, não precisão zero. Passes são intervalos de frames absolutos com fim exclusivo; recuperações têm time e tempo de referência. O pareamento reutiliza máxima cardinalidade/mínimo erro do módulo existente, com tolerância declarada antes de avaliar.

`forecast_error_scores` avalia previsões contra posições futuras fornecidas **apenas na fase de avaliação**, com contagens por horizonte e referências ausentes. Na execução real, essas posições vêm do próprio tracker: medem consistência da previsão, não acurácia física contra tracking corrigido.

Foram exercitados oito cenários sintéticos e seis configurações de sensibilidade, sem selecionar uma pelo score da partida. O CSV público não foi convertido em anotações artificiais de passe/interceptação. Altura, giro corporal detalhado e bola morta continuam fora do modelo.

### Reprodução e resultados locais

```sh
.venv/bin/python -m scripts.run_collective_experiment \
  --data-dir data --baseline outputs/public-2min-missing-ball-v1 \
  --ground-truth evaluation/gt/BAR-ATH_possession_gt_public.csv \
  --out outputs/collective-pass-nova-execucao
```

Usar destino novo. A execução congela parâmetros e hashes antes de avaliar, preserva o histórico 0–14184 e compara a mesma janela pública 11185–14184. A versão anterior é reproduzida como controle. Assets, resultados e animações reais permanecem fora do Git.

Resultado desta entrega: `outputs/collective-pass-v1-verified/ANALISE.md` e `report.html`. Animação gerada pela skill local: `outputs/simulacoes/posse-coletiva-v1-apresentacao/`. Mostrar a comparação com o modelo anterior e não apenas o modelo novo. Não inferir qualidade na partida inteira a partir deste recorte já consultado.

Verificação final: 108 testes passaram. Os testes novos cobrem passes, recebedores, controle adversário, ausência, disponibilidade, previsões, métricas e equivalência por blocos. A simulação compara gabarito, versão anterior e coletiva; os scores e os limites do recorte constam do relatório local.
