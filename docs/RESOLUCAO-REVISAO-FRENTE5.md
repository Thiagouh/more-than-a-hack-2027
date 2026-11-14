# Resolução da revisão independente — frente 5

Data: **04/11/2026**. Referência preservada: REVISAO-INTEGRAL-FRENTE5.md. Escopo aprovado: correções completas primeiro, sem piloto IDSSE, sem commit/publicação automática. O código oficial de posse não foi modificado.

## Defeitos A1–A9

| Item | Resolução | Evidência permanente |
|---|---|---|
| A1 — perda antes da aquisição | Corrigido. Perda tem parent_event_id, une suporte/pistas transitivos e respeita disponibilidade do pai. Fila entrega pai primeiro | ReviewRegressions.test_A1_parent_point_and_hint_delays_are_transitive; teste de demo/fila |
| A2 — cache transforma sucesso em erro | Corrigido. Destino recusado antes de processar; falha operacional de escrita posterior vira write_failed, preservando sucesso e saída | test_A2_cache_destination_rejected_and_postpublication_fault_is_success; falha de leitura/permissão |
| A3 — cache estruturalmente inválido | Corrigido. Validação de objeto, versão, identidade, conjunto de arquivos e formato dos hashes; corrupção vira miss | test_A3_structural_corruption_is_always_miss; teste existente de arquivo corrompido |
| A4 — fronteiras malformadas | Corrigido. Validadores conferem tipo de cada objeto/lista/item antes de acessar conteúdo; CLI usa diagnóstico controlado | test_A4_all_malformed_collection_boundaries_are_safe; test_cli_references_with_invalid_roots_are_audited_without_traceback |
| A5 — evento duplicado conta duas vezes | Corrigido. Todos os eventos v2 são validados antes da seleção; IDs repetidos e frame bool são recusados | test_A5_duplicate_ids_and_bool_frame_rejected |
| A6 — aceite incompleto de contrato | Corrigido. Consume exige manifesto e valida versão, campos, componentes, limites, atrasos e predecessores | test_A6_contract_and_causal_graph_are_checked; teste dos esquemas |
| A7 — flags omitidas na API | Corrigido. quality_flags obrigatório na API e CSV; célula vazia também é erro | test_A7_omitted_flags_rejected_in_api_and_blank_csv |
| A8 — rejeições fora da auditoria | Corrigido. CLI inicia após parse/destino e antes de leitura; API run inicia antes da configuração/metadados. Estados separados e identidade parcial | test_A8_rejection_is_audited_before_config_and_cli_read; teste CLI de referência inválida |
| A9 — latência ambígua | Corrigido. Erro do instante estimado, atraso de confirmação, atraso de disponibilidade e erro da entrega têm definições distintas | test_A9_timing_error_is_distinct_from_confirmation_and_delivery |

## Riscos B1–B8

| Item | Tratamento | Evidência ou pendência |
|---|---|---|
| B1 — matching guloso | Substituído por máxima cardinalidade, mínimo custo absoluto e desempate determinístico. Candidatos por janela e componentes; fronteiras racionais | MatchingProperties: contraexemplo, limite exato, ordem de entrada e enumeração exaustiva de conjuntos pequenos |
| B2 — integração real | **Pendente de consumidor da frente 4**. Entregues validação completa, fila de teste e drenagem após último frame; nenhuma política de posse/retroatividade final implementada | replay/consume; deliveries.csv; report declara real_front4_integration=false |
| B3 — qualidade na partida | **Pendente de assets e referência apropriada**. Nenhuma avaliação pública/privada ou piloto externo executado | Testes e medições identificados como sintéticos; orientação e bola aérea não foram resolvidas |
| B4 — memória global | Pipeline usa SQLite para eventos, IDs e origem, exportação incremental e visão agregada. API detect continua acumulativa; registro padrão do Detector cresce com IDs, pistas ficam indexadas em memória | Dois padrões de escala 199319×22 e teste de equivalência por blocos; limitações no guia |
| B5 — cache concorrente | Lock POSIX por chave cobre leitores/publicadores; revalidação antes de publicar; entrada válida nunca substituída por outro escritor | Teste com dois processos escritores e leitor concorrente, seguido de nova leitura |
| B6 — dependências excessivas | Regras usam dependências específicas: seleção atual, derivadas, checks de salto do rastro e pai. Feature completa pode ter envelope mais amplo que um evento específico | test_unrelated_old_player_does_not_delay_control; derivadas e atrasos utilizados preservados; fixture causal |
| B7 — fps e frames | Perfis originais explicitam 25 Hz. Outra frequência exige reference_fps ou duração; modo duration converte tempo entre primeira/última amostra | test_duration_and_frame_rate_are_explicit; configuração e atraso efetivo no quality.json |
| B8 — hints_status fora da chave | Corrigido. Status participa da identidade e diferencia relatório/cache | test_hints_status_participates_in_identity; test_cache_overlap_and_hint_status_change |

## Melhorias C1–C9

| Item | Entrega | Limite |
|---|---|---|
| C1 — diagnóstico | Cobertura por time/produtor, flags de jogadores/candidatos, ausência, derivadas, margem/movimento e motivos de inelegibilidade | Contagens de dados/regras, não acerto esportivo |
| C2 — ATD | XML com namespaces, IDs/nomes não vazios quando presentes, conferência de fps/times/dimensões, flag de posição implausível sem recorte | Fixture de esquema validada; asset real e orientação não verificados |
| C3 — proveniência de avaliação | Hashes, código, parâmetros e versão do matching nos relatórios; pares, pass/shot e distribuição de erros | Referência precisa estar previamente alinhada; não cobre todos os toques |
| C4 — robustez numérica/estrutural | Bool recusado, limites numéricos suportados, objetos/listas checados, campos extras em extensions | Valores fora da representação declarada são recusados, sem reparo |
| C5 — contrato publicado | Seis esquemas JSON, fixture causal versionada, flags/tipos de pistas documentados, desempate e origem de recorte | Flags de produtores desconhecidas bloqueiam confirmação individual; tipos de pistas desconhecidos são opacos |
| C6 — visualização | Recortes, eixo absoluto, legenda, origem mista hachurada, indicação fora do intervalo e CSV acessível; demo com perda e atraso | Visão geral pode ter sobreposição; recorte/tabela são necessários para detalhe |
| C7 — publicação | Arquivos listados abaixo; exclusão de dados/outputs conferida; workflow de CI sintética preparado | Nenhum commit ou CI remoto executado; não declarar publicação concluída |
| C8 — fontes externas | Protocolo atualizado para matching v2, identidades e separação de atrasos | **Snapshot revalidado e XMLs pequenos ainda pendentes**, conforme opção de não executar piloto nesta rodada |
| C9 — manutenção | Validação de eventos, dependências, fila, armazenamento, auditoria e visualização em módulos próprios; suíte ampliada com regressões e propriedades | Não reescrever tracking, SmartTag ou máquina de estados de posse |

## Verificação técnica e limites

A execução inicial da suíte tinha 32 testes; **60 testes passaram** na suíte ampliada, registrada em outputs/frente5/validacao-v2/tests.txt. Os testes são sintéticos, sem rede, vídeo, tracking real ou gabarito. Incluem JSON Schema (dependência de desenvolvimento), provas pequenas de matching por enumeração e cache multiprocesso.

A baseline e os hashes oficiais estão registrados em outputs/frente5/revisao-baseline/. Os mesmos hashes são verificados permanentemente na suíte. Sem alterações nos três arquivos oficiais:

- possession_metrics.py: 341bad48904b489d6e875397b27687f5d6a4d22a397f3dc8af3a00d48f16874d
- run_evaluation.py: 1c61564fe52499ba4af3006fb5eb4147b825e92bf516651add80de49d267b19f
- validate_submission.py: 07ac29499acbd337e1dc1de1832d63f6e1fa1d981e22f6ad08212529e2bbee14

Medições computacionais detalhadas e seus snapshots de código estão em outputs/frente5/escala-v2*/ e nos manifestos dos pipelines. Relatório consolidado: DESEMPENHO-FRENTE5.md. São medições de uma máquina e cenários construídos, sem garantia de tempo/memória para qualquer arquivo nem alegação de qualidade esportiva. O custo do matching externo depende dos componentes de ações dentro das janelas; ele não foi medido em um dataset externo.

## Arquivos a revisar para versionamento

Lista explícita da implementação e da infraestrutura desta rodada:

```text
.github/workflows/frente5.yml
requirements-dev.txt
src/README.md
src/ball_events/__init__.py
src/ball_events/__main__.py
src/ball_events/artifacts.py
src/ball_events/audit.py
src/ball_events/contract.py
src/ball_events/demo.py
src/ball_events/dependencies.py
src/ball_events/detector.py
src/ball_events/evaluation.py
src/ball_events/events.py
src/ball_events/inputs.py
src/ball_events/storage.py
src/ball_events/visual.py
starter/load_atd.py
configs/README.md
configs/frente5_v1.json
configs/frente5_dois_frames.json
schemas/envelope-v1.schema.json
schemas/event-v2.schema.json
schemas/hint-v1.schema.json
schemas/manifest-v1.schema.json
schemas/player-v1.schema.json
schemas/point-v1.schema.json
scripts/benchmark_frente5.py
tests/test_ball_events.py
tests/test_ball_events_v2.py
tests/fixtures/ball_events/causal-v2.json
docs/FRENTE5-EVENTOS.md
docs/RESOLUCAO-REVISAO-FRENTE5.md
docs/DESEMPENHO-FRENTE5.md
docs/AVALIACAO-EQUIPE.md
docs/DADOS-EXTERNOS-FRENTE5.md
```

Dependências locais anteriores ainda não versionadas, preservadas e necessárias para clone limpo e suíte completa:

```text
evaluation/comparar_csv.py
evaluation/desenhar_posses.py
scripts/demo_timeline.py
tests/test_avaliacao.py
tests/test_timeline.py
```

O documento independente docs/REVISAO-INTEGRAL-FRENTE5.md permanece preservado como referência; pode acompanhar a revisão conforme decisão da equipe. Não incluir outputs/, assets ou manifestos reais. Não executar git add indiscriminadamente. Esta lista não implica stage, commit ou publicação.
