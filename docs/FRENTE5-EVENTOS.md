# Frente 5 — eventos v2 e validação

Atualização de **04/11/2026**, após a revisão independente. Entradas continuam em v1; eventos passam a v2. O módulo entrega evidências à frente 4 e não implementa posse final. H = Barcelona; A = Athletic Club. Força low/medium/high é ordinal, sem calibração probabilística.

## Começar e revisar

Execute da raiz do repositório. O ambiente é `.venv` dentro dessa raiz, não o homônimo da pasta superior.

```sh
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m src.ball_events demo --out outputs/frente5/minha-demo-v2
.venv/bin/python -m src.ball_events replay --run outputs/frente5/minha-demo-v2/run --drain --out outputs/frente5/minha-fila-v2
.venv/bin/python -m src.ball_events render --run outputs/frente5/minha-demo-v2/run --start 0 --end 16 --out outputs/frente5/meu-recorte-v2
```

Use diretórios novos: saídas existentes não são sobrescritas. A demo contém aquisição atrasada, perda dependente, toque, disputa, bola estimada e ausente. Todos são construídos por código. Abra `events.svg` e `events-table.csv`; o segundo oferece leitura sem dicas por mouse. A demo torna explícito que perda e aquisição podem ficar pendentes depois do último frame e ser entregues na drenagem posterior.

`jsonschema` é dependência **de desenvolvimento**, para conferir os esquemas publicados; não é importado pelo runtime. Execução normal continua usando biblioteca padrão, NumPy e pandas. O cache usa locks POSIX, verificados em macOS e por teste preparado para Linux; não declarar suporte de cache no Windows sem adaptar/testar o lock.

## Interfaces e compatibilidade

Manifesto, ponto, jogador, envelope e pista estão formalizados em `schemas/*-v1.schema.json`. Eventos estão em `schemas/event-v2.schema.json`. Os esquemas descrevem estrutura; validadores Python acrescentam coerência entre campos, limites da timeline, segmentos e dependências causais. A fixture de referência está em `tests/fixtures/ball_events/causal-v2.json` e é validada pelos dois mecanismos.

Manifesto mínimo para uma partida completa:

```json
{
  "contract_version": "v1",
  "dataset_id": "meu-dataset",
  "timeline": "tactical",
  "fps": 25,
  "n_frames": 199319,
  "pitch_m": [105.3, 68.0],
  "coordinates": "metres_corner_fixed",
  "teams": {"H": "Barcelona", "A": "Athletic Club"},
  "producers": {"frente1": "1", "frente2": "1"},
  "continuity_starts": [],
  "files": {"players": "players.csv", "ball": "ball.csv", "hints": "hints.jsonl"}
}
```

Paths são relativos à pasta do manifesto. Timelines aceitas: tactical, synthetic e external; broadcast precisa de adaptação explícita para a referência apropriada. Metros, origem no canto superior esquerdo da representação do campo, x ao longo do comprimento e y ao longo da largura, eixos fixos ao campo. Não inverter por ataque. Para recortes, declare `is_clip: true` e `origin: {"timeline":"tactical","first_frame":1000,"fps":25}`. O frame local f corresponde a `first_frame + f * origin.fps / fps`; não é necessário arredondar essa relação. Os nomes dos datasets não substituem essa declaração.

CSVs exigem `frame,x_m,y_m,source,producer,quality_flags`; jogadores acrescentam `track_id,team`. A bola tem exatamente uma linha por frame, jogadores podem ter zero ou várias, ordenadas por frame e com IDs únicos por frame. `quality_flags` deve conter lista JSON explícita, por exemplo `[]`: omissão na API ou célula vazia no CSV são erros.

`source`: observed, interpolated, estimated, missing ou unknown. Para missing, ambas as coordenadas são vazias no CSV e null na API; as demais fontes exigem números finitos suportados. Unknown nunca confirma eventos. Observed significa observação do tracker, não posição verificada.

Opcionais: `support_start_frame`, `support_end_frame` e `available_at_frame`; defaults são frame, frame+1 e fim do suporte−1. O fim é exclusivo. Suporte não atravessa limites conhecidos de continuidade. Disponibilidade pode ultrapassar a timeline. Ausência de pontos ou flags não é certeza de qualidade.

Produtores são declarados com versão e IDs sem `:`. O detector acrescenta namespace ao rastro original (`producer:track_id`). Um rastro não é identidade pessoal nem camisa; mudança de time exige novo ID. Campos extras pertencem a um objeto `extensions`; não entram automaticamente na evidência. Inteiros suportados são 0..2^53−1, sem bool. Números devem ser finitos, com módulo até 1e100; parâmetros estritamente positivos devem ser pelo menos 1e-100; limites físicos são tratados separadamente por flags/regras.

Pistas JSONL:

```json
{"hint_id":"restart-1","type":"restart_hint","start_frame":10,"end_frame":20,"team":null,"strength":"low","available_at_frame":22,"producer":"frente3"}
```

O produtor deve constar no manifesto. IDs de pistas são únicos. Tipos conhecidos para documentação: restart_hint e stoppage_hint; outros tipos são preservados como contexto opaco, sem nova regra futebolística. Pistas não produzem D automaticamente. Se não houver arquivo opcional, registrar not_provided/optional_file_missing; não interromper por isso.

API: `detect(manifest, frames, config=None, hints=())` devolve features, eventos recém-confirmados e qualidade. O envelope é `{"frame": f, "ball": ponto, "players": [pontos]}`. `Detector.feed()` permite alimentar blocos mantendo a mesma instância; `finish()` exige cobertura integral. Os objetos recebidos não são alterados. `validate_events(events, manifest)` valida o contrato e os pais; `consume(events, manifest)` e `replay(events, manifest, drain=False)` verificam entrega. Não existe aceite completo de timeline sem manifesto.

Eventos antigos v1 devem ser regenerados. O validador não inventa pais para perdas de arquivos antigos. As entradas v1 foram mantidas, mas validações anteriormente permissivas agora recusam erros. Outputs antigos são preservados e o novo cache tem identidade própria.

## Regras e causalidade

Os thresholds espaciais e de movimento permanecem os originais, ainda não validados na partida: proximidade/perda 2 m, controle 1,5 m, margem 1 m, disputa 2 m com diferença ≤1 m, velocidade relativa ≤5 m/s, mudança de velocidade ≥3 m/s ou ângulo ≥30° com velocidades ≥2 m/s, salto de bola >45 m/s, jogador >12 m/s, duplicatas ≤0,3 m.

- Toque: proximidade no frame central e mudança de movimento; confirma no próximo. Proximidade sozinha não basta.
- Aquisição: mesmo rastro próximo, margem e movimento relativo compatível durante a persistência.
- Perda: afastamento persistente depois de aquisição candidata; contém obrigatoriamente `parent_event_id` dessa aquisição.
- Disputa: evidência coletiva por episódio, sem vencedor. Interpolação/estimativa só permite proximidade descritiva e disputa fraca.

Ausência, salto, ambiguidade e limite de continuidade suspendem derivadas e memória individual relevante. Não produzir perda por oclusão ou troca de slot. Não somar duplicatas como reforço. Bola imóvel não vira D. Não distinguir bola aérea usando apenas XY. Flags de produtores continuam bloqueando confirmação individual enquanto o vocabulário não estiver alinhado.

Cada operação registra dependências: seleção examina os jogadores do frame; derivadas usam observações específicas da bola; movimento relativo e saltos usam os rastros envolvidos. Um jogador distante **atual** pode ser necessário para provar quem é o mais próximo. Sua observação histórica não atrasa automaticamente uma regra que não a utiliza.

Perdas carregam transitivamente suporte, pistas e disponibilidade da aquisição. Portanto, `available_at_frame(perda) >= available_at_frame(aquisição)`. Se os dois ficam disponíveis juntos, a aquisição é entregue primeiro. Pais ausentes, tipo errado, rastro/time/segmento diferentes, omissões de suporte e ciclos são recusados.

Cada evento inclui versão v2, ID, tipo, time opcional, rastros, componentes, razões, flags, segmento, frame do evento, confirmação, suporte, disponibilidade, lookahead, atraso e dependências. `context_hints` é preservado inclusive transitivamente. Os IDs são determinísticos.

### Persistência e frequência

Os arquivos `configs/frente5_v1.json` e `configs/frente5_dois_frames.json` mantêm nomes históricos; **ambos agora geram eventos v2**. Declaram modo frames e reference_fps=25.

Três amostras consecutivas a 25 Hz confirmam 80 ms depois do início; duas confirmam 40 ms depois. Contar amostras e medir tempo entre primeira/última observação são coisas diferentes. O primeiro deslocamento aquece a derivada: numa sequência estável iniciada em zero, o default começa em 1 e confirma em 3.

Frequência diferente exige um perfil explícito. Alternativas:

```json
{"persistence_mode":"frames","reference_fps":10,"persistence_frames":3}
```

```json
{"persistence_mode":"duration","persistence_seconds":0.12}
```

Duração significa tempo mínimo entre evento e confirmação. Amostras efetivas = `max(2, ceil(seconds * fps) + 1)`. Não misturar persistence_seconds com persistence_frames/reference_fps. A configuração resolvida no manifesto é informativa; para repetir um modo de duração, use os parâmetros de entrada desse modo, sem acrescentar o campo derivado persistence_frames. O relatório registra modo, valor e atraso físico efetivo. Não escolher o perfil pela janela pública.

## Executar com entradas próprias ou ATD

```sh
.venv/bin/python -m src.ball_events detect --input caminho/manifest.json --config configs/frente5_v1.json --hypothesis 'Conferir contrato e cobertura sem alegar qualidade' --cache-dir outputs/frente5/cache --out outputs/frente5/minha-execucao
.venv/bin/python -m src.ball_events detect --source atd --config configs/frente5_v1.json --hypothesis 'Conferir leitura e cobertura local' --out outputs/frente5/meu-atd
```

A hipótese é registrada antes do processamento. Não escrever anotações da partida nesse campo. Use `--data-dir` ou MTAH_DATA_DIR para localizar a pasta que contém tactical/. Só ATD é necessário; não são requeridos vídeo, SmartTag ou homografia. O leitor reaproveita resolução de assets e converte frame_id−1, sem offsets do gabarito. Converte coordenadas normalizadas pela dimensão aplicada de games.py.

O leitor reconhece namespaces XML, confere IDs, nomes quando presentes, times, slots e frequência declarada. Diferença de campo de até 1 m é registrada (incluindo 105 contra 105,3); diferença maior exige correção explícita. Coordenadas implausíveis são sinalizadas, sem recorte. Ordem H/A segue os metadados e é conferida por nomes quando fornecidos. Sem asset real, isso não comprova orientação nem qualidade do tracking.

## Artefatos, fila e inspeção

| Arquivo | Uso |
|---|---|
| features.csv | Uma linha/frame: distâncias, margens, movimento, origem/qualidade e suporte das features |
| events.jsonl | Eventos v2 em ordem de entrega, com predecessores antes dos dependentes |
| quality.json | Cobertura por time/produtor, fontes, flags, motivos de inelegibilidade, lacunas e reinícios |
| context_hints.json | Pistas preservadas |
| sources.csv | Intervalos de origem da bola, para renderização local |
| events.svg | Visão geral; pixels com múltiplas origens ficam hachurados |
| events-table.csv | Tabela acessível de eventos, pais, frames absolutos e atrasos |
| consumer_check.json | Aceite de contrato/fila; declara que não é integração real com a frente 4 |
| deliveries.csv | Entregas e eventos pendentes no fim |
| manifest.json | Hashes, configuração resolvida, versões e identidade do cache |
| experiment.json | Hipótese, comando e limitações sem horários operacionais |

`feed()` devolve confirmações para processamento offline. O consumidor deve aguardar available_at_frame. A simulação recebe em ordem de confirmação e usa fila para liberação; não transforma evidência retroativa em correção de labels. Sem `--drain`, o fim da timeline conserva pendências. A drenagem permite inspecionar entregas posteriores, mantendo registrado o que estava pendente no fim original. O pipeline usa fila/ordenação em SQLite e exporta detalhes sem reter todos os eventos em listas.

Na visão completa, marcas podem se sobrepor. Use `render --start F --end G` (limites locais, fim exclusivo). Eixo mostra frames absolutos da origem; a tabela mostra o mapeamento. Disponibilidade fora do intervalo recebe seta e texto explícito. A renderização não consulta vídeo nem gabarito.

## Auditoria e cache

Uma tentativa começa após parse dos argumentos e identificação de comando/destino, antes de ler arquivos ou validar configuração. A API run começa antes da validação semântica, dado um destino de saída válido. Ajuda e erros de sintaxe de argumentos ficam fora das tentativas experimentais.

Registros em `_experiments/`, ao lado do destino, distinguem started, rejected, failed_processing, failed_publication e completed. Identidade pode ser parcial numa rejeição. Registram código/etapa do erro, não registros brutos. Horários UTC ficam nesses registros operacionais, fora dos artefatos determinísticos. Report-csv, evaluate-events, render e replay também registram tentativas/proveniência.

Saída final só aparece depois de processamento/verificação completos. Destino de cache inválido é recusado previamente. Cache é opcional, descartável e bloqueado por chave com lock do sistema operacional. Cada leitura confere tipos, versão, identidade, lista exata de arquivos e hashes; corrupção ou falha de cópia causa recálculo. Um cache válido publicado por outro processo é reutilizado, nunca removido por outro escritor.

Após publicação íntegra, falha ao gravar cache retorna sucesso com cache_status=write_failed e aviso, registrado na tentativa. Não há falha falsa da execução principal. Cache, inputs e output não podem se sobrepor. A identidade inclui hints_status, código inclusive não versionado relevante, esquemas, scripts, entradas, parâmetros e dependências. Na API run, o produtor é responsável por fornecer caminhos que identifiquem realmente os dados lidos por frames_factory.

## Avaliação

```sh
.venv/bin/python -m src.ball_events report-csv caminho/submissao.csv --rows 199319 --out outputs/frente5/relatorio
.venv/bin/python -m src.ball_events evaluate-events --events outputs/frente5/minha-execucao/events.jsonl --fps 25 --reference caminho/referencia.json --out outputs/frente5/avaliacao.json
```

Evaluate-events usa o manifest.json ao lado de events.jsonl; `--manifest` permite informar outro caminho. A frequência deve coincidir. Exige eventos v2 completos, IDs únicos e pais válidos antes de selecionar toques. Referência contém metadata com provenance/independence/timing e events com reference_id/type (pass/shot)/team/time_s. Todos os tempos devem pertencer à mesma referência temporal previamente alinhada. Não ressincronizar usando o detector.

Matching próprio v2: maximizar número de pares, depois minimizar soma do erro absoluto e resolver empates por ordenação canônica estável de IDs. Janelas temporais e componentes bipartidos evitam criar todas as combinações fora da tolerância; custos racionais preservam fronteiras decimais. Tolerâncias fixas: 0,2/0,5/1,0 s. Exporta pares, referências não cobertas e predições sem referência, geral e por pass/shot. Duas referências com IDs distintos não autorizam contar duas cópias de uma mesma predição.

Métricas separadas: event_time_error_s (evento−referência, com sinal); confirmation_delay_s (confirmação−evento); availability_delay_s (disponibilidade−evento); delivery_time_error_s (disponibilidade−referência); erro absoluto de localização. Cada uma tem count, cobertura dos campos entre pares, média, mediana e quantis p05/p90/p95 por interpolação linear na posição (n−1)*q. Reference_recall é a fração de referências cobertas; não confundir com cobertura dos campos. Predições sem referência não são falsos positivos automaticamente. Não há precisão/F1 de todos os toques nem validação de controle contínuo.

O relatório individual reaproveita a validação oficial e resumo existentes. X interrompe blocos no resumo bruto; não equivale ao preenchimento temporal oficial. Comparação e timeline H/A/D/X existentes permanecem disponíveis. O avaliador oficial de posse não foi alterado, inclusive seu matching.

Conferência pública futura usa configuração congelada e apenas um CSV escolhido da frente 4. Não selecionar parâmetros/versões pelo score público, número de posses ou percentuais D/X. Ablação requer o mesmo consumidor real, mesmas entradas/código/configuração, variando somente os eventos.

## Desempenho e limites de aceite

```sh
.venv/bin/python scripts/benchmark_frente5.py --out outputs/frente5/escala-churn
.venv/bin/python scripts/benchmark_frente5.py --pattern events --out outputs/frente5/escala-eventos
```

São 199.319 frames ×22 jogadores, IDs trocados a cada 250 frames. Churn alterna observed/estimated a cada frame; não produz eventos porque interrompe a confirmação observada. Events usa bola observada com movimentos e pausas, exercitando aquisições, perdas e toques. O script mede detector e pipeline em processos separados, parsing/escrita/exportação, geração dos CSVs separadamente, pico RSS e armazenamento temporário. Inputs sintéticos grandes são removidos ao fim.

Detector.feed mantém histórico limitado, mas o registro padrão de IDs cresce O(IDs) e as pistas/indexação O(pistas). Detect() acumula features/eventos. A execução de arquivos usa registro com cache limitado e SQLite para IDs/eventos/origens; SVG e CSVs são escritos incrementalmente. O tamanho dos artefatos e os componentes do matching externo ainda crescem com a quantidade de dados; não alegar memória constante para toda API ou avaliação.

A matriz de correções está em RESOLUCAO-REVISAO-FRENTE5.md. Os testes demonstram contrato e comportamento sintético. Dados da partida e consumidor real permanecem indisponíveis nesta entrega, assim como o piloto externo não executado. Bola aérea, ambiguidade de tracking e força heurística continuam limitando conclusões esportivas.

Não produzir novas anotações manuais nem enviar dados reais a serviços. Outputs, assets e manifestos reais permanecem fora do Git público. Nenhum commit/publicação é automático. Divergências de regulamento/edição continuam em AVALIACAO-EQUIPE.md; pesquisa e snapshot externo estão em DADOS-EXTERNOS-FRENTE5.md.
