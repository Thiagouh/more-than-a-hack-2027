# Posse coletiva: kickoff e recepção provável

A política `collective-pass-v2-experimental` mantém o detector de eventos e o consumidor anterior intactos. O catálogo de métricas passa a `collective-metrics-v2`; candidatos de encontro incluem `distance_m`.

## Kickoff

`CollectiveModel(meta, ..., kickoffs=())` aceita registros com `half` (1 ou 2), `frame`, `team` (H/A), `available_at_frame` e `source_id`. Há no máximo um kickoff por tempo. Não é um evento de aquisição de controle individual.

No frame indicado, uma marcação já disponível limpa o histórico coletivo. A bola precisa estar observada e válida, com jogador observado e sem alertas do time indicado a até `proximity_m`, e o candidato adversário ao menos `margin_m` mais distante. Nesse caso a posse começa imediatamente, com motivo `kickoff_tracking_confirmed`. Conflito ou ausência de tracking preservam a inferência normal. Marcações atrasadas são diagnosticadas e não aplicadas retroativamente. Os eventos do detector continuam idênticos com ou sem kickoffs.

`load_kickoffs(data_dir, meta)` lê apenas `Kick Off (Start)` do Smart Tagging, associa o time pelo nome e usa os frames exatos publicados em `starter/load_smart_tagging.py`. O início do bloco de tracking do segundo tempo não é o kickoff. Tags ausentes, ambíguas ou incompatíveis não inventam um time. A disponibilidade declarada corresponde ao instante do kickoff para esse uso offline explícito; uma integração online deve fornecer sua disponibilidade real.

## Recepção provável sustentada

Um mesmo jogador adversário deve satisfazer por 0,20 s consecutivos: bola observada e válida, cobertura de movimento dos dois times, erros de trajetória instantâneo e RMS de até 0,75 m, bola em movimento a pelo menos 2 m/s, encontro em até 0,60 s, vantagem de chegada de pelo menos 0,20 s, aproximação prevista de até 1,5 m e distância atual diminuindo. Tempo de chegada adversário indisponível não significa vantagem infinita.

Os parâmetros novos são `reception_sustain_s`, `reception_horizon_s` e `reception_advantage_s`. O raio, cobertura e limites de trajetória reutilizam os parâmetros físicos existentes. A duração conta frames elegíveis / fps (cinco frames a 25 fps), após haver observação anterior do mesmo candidato para comprovar aproximação.

Troca de candidato, interrupção ou falha de qualquer requisito limpa a sequência. Aquisição confirmada tem prioridade. A aceitação muda apenas a posse coletiva, com estado `recepcao_provavel` e motivo `sustained_probable_reception`, sem criar evento de controle. O suporte acumula as dependências da sequência.

Cada snapshot inclui `kickoff` e `reception`, com motivo, candidato, duração e verificações individuais. Resultados registram versões, parâmetros e kickoffs; o experimento registra também a identidade do arquivo de marcações quando disponível. O gabarito público é lido somente depois das previsões, para avaliação.

## Verificação

Execute `.venv/bin/python -m unittest discover -s tests` na raiz do projeto. Os testes incluem kickoff dos dois tempos, conflito, ausência, atraso, equivalência por blocos, preservação dos eventos, recuperação antecipada e quebra dos requisitos individuais.

## Continuação: kickoff sem equipe na tag e ponte visual (v3)

A política atual é `collective-visual-restarts-v3-experimental`. O mecanismo v2 existia, mas não inicializava a execução real: o arquivo de Smart Tagging estava ausente localmente e, após recuperado da pasta autorizada, mostrou `Team=N/A` nos dois inícios de tempo. O loader antigo descartava essas marcações. Além disso, a bola não é observada nos primeiros frames do kickoff.

O loader agora preserva a âncora de início mesmo sem equipe (`team=null`). **Somente no frame de um kickoff explicitamente marcado**, uma equipe com jogador observado a até 2 m do centro e adversário observado pelo menos 3 m mais distante pode inicializar a posse. Pontos indisponíveis, saltos e duplicatas são excluídos. Ausência de adversário ou presença ambígua no centro não decide equipe. Não se usa cor do uniforme, gabarito de posse ou tags de fases para preencher a equipe. A hipótese recebe motivo `kickoff_center_setup_inferred`; não cria um evento de aquisição individual nem uma posição artificial de bola. O caminho antigo com equipe conhecida e bola observada permanece distinto (`kickoff_tracking_confirmed`).

Essa regra é uma hipótese geométrica de formação inicial, não prova de toque ou controle. Não vale em qualquer passagem da bola pelo círculo central. Mantêm-se expiração por falta de bola, conflitos e disponibilidade causal. O detector de eventos continua inalterado, preservando o modelo anterior como comparador.

### Controle visual integrado à posse

`CollectiveModel.feed(envelope, visual_observations=())` aceita observações SAM em uma via separada do tracking. A ponte em `src/team_possession/visual_control.py` exige imagem vinculada por hash, equipe/papel, execução SAM explícita, máscara elegível, contexto válido e evidência de controle. Identificar a equipe do goleiro **não basta** para atribuir posse.

- `reviewed_held_ball`: confirmação manual de bola nas mãos de um goleiro, com revisor explícito. Pode sustentar o estado do frame atual, com motivo `visual_reviewed_goalkeeper_control`.
- `ball_body_proximity`: evidência automática de proximidade entre bola e corpo exige três frames consecutivos elegíveis da mesma equipe antes de sustentar `visual_control_corroborated`. O produtor precisa realmente detectar a bola; a ponte não cria essa detecção. O executor real de cena descrito abaixo ainda utiliza a revisão manual das mãos, não esse detector automático.
- Sem controle, máscara inválida, contexto desconhecido, equipes visuais conflitantes, observação atrasada ou conflito com controle observado no tracking: nenhuma inicialização visual. Aquisição do tracking e kickoff têm prioridade. A evidência atrasada não reescreve frames anteriores.

As posições e identidades ATD não são alteradas. Não se desenha nem se cria bola ausente para sustentar a hipótese. O snapshot registra `visual_control`, fontes e dependências; o suporte automático guarda somente os três frames de corroboração para limitar memória. São heurísticas experimentais, sem probabilidades calibradas.

### Executor SAM real e teste de cena

```sh
.venv-sam3-mac/bin/python -m scripts.run_sam3_goalkeeper \
  --manifest outputs/sam3-goalkeeper-scene-input/scene.json \
  --checkpoint data/sam3-checkpoint/sam3.pt \
  --checkpoint-sha256 9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e \
  --out outputs/sam3-goalkeeper-real-next

.venv/bin/python -m scripts.run_visual_scene_bridge \
  --package outputs/sam3-goalkeeper-real-next/visual-observations.json \
  --out outputs/sam3-goalkeeper-possession-next
```

O executor usa CPU, um modelo/processo, watchdog de 60 s e RSS amostrado limitado a 7 GB. Mantém checkpoint, prompt `person`, seleção baseline e binarização 0,5; não explora novos limiares SAM. A caixa do goleiro e o contexto do lance são revisados, não descobertos automaticamente. O descritor v3 e o vazamento da máscara são verificados. Uma heurística de cor provisória exige pelo menos 60% dos pixels do núcleo compatíveis com verde/azul-esverdeado (G > 1,12 R; G >= 0,75 B; máximo >50; amplitude >15% do máximo). Esse filtro não foi calibrado e não comprova isolamento do tronco nem distinção universal de grama/uniforme. Seu papel é evitar promover contexto manual quando o SAM não fornece máscara utilizável e indício de uniforme compatível; não autoriza congelar uma referência.

Teste real em `outputs/sam3-goalkeeper-real-bridge/`: quadro RGB que mostra o goleiro verde com bola nas mãos aos 00:18, amostra 31 da gravação. A inferência concluiu em **24,75 s**, com RSS máximo amostrado de **6.842.826.752 bytes**, núcleo com **81,37%** de pixels compatíveis e decisão contextual H. A inspeção de `sam-review.png` mostra a silhueta do goleiro isolada nesse caso. Isso não aprova as 22 máscaras iniciais nem constitui avaliação de precisão.

A posse foi reexecutada sobre essa cena em `outputs/sam3-goalkeeper-possession-scene-v3-final/scene-bridge.json`: **X sem observação visual → H com SAM + contexto + bola nas mãos revisada**. Não houve coordenadas de bola/jogadores fabricadas ou eventos artificiais do detector. Trata-se de um frame externo, com origem e controle manual explicitados, não de uma execução automática sobre os 120 s tactical.

### Aplicação aos mesmos frames tactical

O experimento principal aceita agora:

```sh
.venv/bin/python -m scripts.run_collective_experiment \
  --data-dir data --baseline outputs/public-2min-missing-ball-v1 \
  --ground-truth evaluation/gt/BAR-ATH_possession_gt_public.csv \
  --visual-observations CAMINHO/visual-observations.json \
  --visual-video data/tactical/BAR-ATH_tactical.mp4 \
  --out outputs/collective-visual-next
```

O pacote deve declarar a timeline tactical, frame do próprio vídeo e `same_source_video_frame=true`, com `source_video_sha256` igual ao arquivo fornecido. O loader valida os hashes do vídeo, da imagem e das observações, os limites do trecho e a disponibilidade. Pacotes externos/broadcast são recusados. Um pacote de SAM de uma gravação de tela não pode ser reaproveitado como se `31 s` significasse `frame tactical 31`. Essas declarações exigem extração/revisão correta da fonte; hashes sozinhos não demonstram a verdade de uma sincronização declarada.

O vídeo tactical e a homografia não estão disponíveis localmente neste estado; a tentativa pelo conector não materializou esses arquivos. Por isso, **SAM/goleiros estão conectados ao motor e exercitados numa cena real, mas ainda não foram aplicados aos mesmos 120 s tactical**. O reconhecimento automático da bola e a revisão de referências de linha também permanecem pendentes. Nenhuma saída antiga foi rebatizada como inferência visual completa.

### Nova execução real de kickoff

`outputs/collective-visual-restarts-v3/` foi recalculado com o prefixo causal até o frame 14184, sem observações visuais tactical. O primeiro frame público (11185) passou a **Athletic/A**, com bola ausente e fonte `kickoff_center_setup_inferred`. O comparador anterior continua X nesse frame. A execução registra o hash do Smart Tagging realmente carregado. Nesse recorte, X caiu de **922 para 834 frames** e a acurácia pública passou de **62,30% para 63,87%**; são métricas exploratórias do mesmo trecho público já consultado, não validação independente nem efeito do SAM. Não houve seleção/ajuste dos parâmetros pelo gabarito.

A nova simulação de conferência fica em `outputs/simulacoes/tracking-gabarito-kickoff-v3/`, com gabarito, comparador e modelo coletivo. A legenda informa que as observações visuais ainda não foram aplicadas nessa execução tactical.

### Proteção do passe de abertura e resultado conferido

A conferência frame a frame detectou que inicializar apenas o instante ainda devolvia X durante o aquecimento de trajetória e o primeiro passe. A hipótese agora pode permanecer no estado explícito `kickoff` / motivo `kickoff_opening_prior`, pelo limite **já existente** de `max_pass_without_control_s` (4 s no perfil atual). Não atualiza artificialmente a idade do último controle. Termina antes diante de controle observado, aquisição do tracking, controle visual, recepção, adversário próximo da bola ou toque adversário, bola fora do campo ou ausência acima da tolerância de 5 frames. Ausência curta da bola não é confundida com uma observação inválida que cancelaria a hipótese imediatamente. Não se extrapola esse comportamento para outras passagens pelo centro.

O snapshot registra `kickoff_opening_prior` com equipe, frame e fonte. A abertura é hipótese de equipe, não evento de controle individual. A duração e a tolerância são as do perfil registrado, sem escolha pelo score público. Casos sintéticos cobrem o atraso no aparecimento da bola, o término no limite e o cancelamento por adversário próximo, além de inicializações independentes dos dois tempos.

A execução final conferida está em **`outputs/collective-visual-restarts-v3-validated/`** e substitui, para análise atual, os snapshots intermediários `v3` e `v3-final`. No mesmo recorte já explorado, X passou de 922 para **775 frames**, e a acurácia pública de 62,30% para **65,83%**. A abertura não some após os dois primeiros frames: há 59 frames adicionais com hipótese explícita de kickoff. Esses números são diagnóstico exploratório da correção, sem validação independente e sem efeito de observações SAM tactical.

A simulação correspondente é **`outputs/simulacoes/tracking-gabarito-kickoff-v3-validado/`**. O teste externo final do goleiro, reexecutado com o motor atual, está em **`outputs/sam3-goalkeeper-possession-v3-validated/scene-bridge.json`**, preservando X → H e a indicação de bola nas mãos revisada manualmente.

## Continuação com o vídeo original do Drive — integração tactical em três frames

O vídeo original foi obtido por download autenticado iniciado pelo usuário e copiado para `data/tactical/BAR-ATH_tactical.mp4`. O conector recusava o arquivo de 5.222.307.334 bytes por seu limite de 256 MB; a homografia pôde ser recuperada pelo conector e está em `data/tactical/BAR-ATH_tactical_homography.csv`. **Esses dois arquivos agora estão disponíveis localmente.** Proveniências privadas: `video-source.json` e `homography-source.json`, na mesma pasta. Nenhum dado foi publicado ou adicionado ao Git.

Foram conferidos 1920×1080, 25 fps e 199.319 frames no MP4; a homografia tem 199.319 linhas contíguas, a partir de zero. `scripts/prepare_tactical_visual.py` extrai pelo contador original decodificado, sem escalonar ou alterar fps, verifica dimensões/contagem/frequência/tamanho do original e registra o SHA256 do vídeo e de cada PNG. Um rascunho mantém caixas/contexto como não revisados. Pontos ATD só são projetados se observados, sem flags e já disponíveis no frame; não se usam identidades/equipes desses pontos para rotular visualmente o goleiro.

```sh
.venv/bin/python -m scripts.prepare_tactical_visual \
  --video data/tactical/BAR-ATH_tactical.mp4 \
  --frames 11635 11689 11690 11691 11710 \
  --out outputs/tactical-visual-frames-next
```

A extração final usada está em `outputs/tactical-visual-frames-reviewed-source/`. Os frames 11689–11691 mostram o goleiro verde segurando a bola diante do gol esquerdo, no primeiro tempo; o placar visível está próximo de 00:19. **20,16–20,24 s são posições relativas no recorte, não segundos convertidos da gravação de tela ou do placar.** As caixas/contextos revisados estão em `outputs/tactical-goalkeeper-scenes/`. O pixel da bola do ATD, projetado, caiu por volta de (437,345), enquanto a bola visível nas mãos aparece perto de (401,302). Essa projeção foi reprovada como evidência visual de controle. A causa não foi atribuída automaticamente a um offset: altura da bola/erros de tracking/projeção ainda precisam ser investigados; nenhum deslocamento foi inventado para ajustar a figura.

### SAM real, revisão e resultado negativo da bola automática

O executor `scripts/run_sam3_goalkeeper.py` agora também pode usar `detect_ball=true`: reutiliza o mesmo backbone/modelo, reseta os prompts e consulta `soccer ball`, com confiança e binarização oficiais inalteradas. A associação exige candidato pequeno, aproximadamente compacto e junto à região superior da pessoa. O teste real em `outputs/tactical-sam3-ball-11689/` **não produziu bola aceitável**. Não houve redução de limiar ou busca de prompts até aprovar. O filtro/projeção em `src/team_identification/visual_geometry.py` tem testes, mas não está validado como detector de bola no jogo.

O SAM identificou a aparição do goleiro nos três frames. A identidade contextual H foi mantida após inspeção do RGB, máscara e núcleo. As máscaras têm halo superior, que fica registrado; os núcleos usados nesse teste contêm camisa do próprio goleiro e a região da bola nas mãos, distintos do tronco do atleta branco adjacente. **Aprovação da aparição experimental não é aprovação de template de referência.** As 22 referências anteriores seguem reprovadas e nenhuma nova referência de linha foi congelada.

`scripts/review_sam3_observation.py` aplica revisão posterior vinculada ao hash do pacote, imagem e máscara, preservando o candidato original. O loader de posse exige `mask_reviewed=true`, revisor e hash da máscara para entradas tactical, além de vincular a observação ao frame declarado pelo pacote. A revisão não remove bloqueadores geométricos. Uma eventual bola SAM aceita exige revisão específica de seu próprio asset. `held_ball_verified=true` representa **confirmação manual de bola nas mãos** por revisor identificado; altera explicitamente o modo para `reviewed_held_ball`, não para detecção automática. Essa anotação precisa estar verdadeira na imagem, não apenas na equipe prevista.

```sh
.venv-sam3-mac/bin/python -m scripts.run_sam3_goalkeeper \
  --manifest outputs/tactical-goalkeeper-scenes/00011689.json \
  --checkpoint data/sam3-checkpoint/sam3.pt \
  --checkpoint-sha256 9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e \
  --out outputs/tactical-sam3-next

.venv/bin/python -m scripts.review_sam3_observation \
  --package outputs/tactical-sam3-next/visual-observations.json \
  --review CAMINHO/REVISÃO_VINCULADA_AOS_HASHES.json \
  --out outputs/tactical-sam3-next-reviewed
```

O executor congela hashes de entrada, política, RGB e código antes de inferir e recusa mudanças durante a execução. O primeiro pacote foi regenerado com essa proteção, conservando os mesmos bytes de máscara já inspecionados. Os pacotes finais são `tactical-sam3-11689-final-reviewed`, `tactical-sam3-11690-reviewed` e `tactical-sam3-11691-reviewed` em `outputs/`. Os candidatos/negativos anteriores continuam preservados. Inferências finais: 28,33 s / 6,05 GB, 25,39 s / 6,07 GB e 25,15 s / 6,70 GB de RSS amostrado, respectivamente, sempre CPU, um processo por vez e limites de 60 s / 7 GB.

### Execução integrada conferida

```sh
.venv/bin/python -m scripts.run_collective_experiment \
  --data-dir data --baseline outputs/public-2min-missing-ball-v1 \
  --ground-truth evaluation/gt/BAR-ATH_possession_gt_public.csv \
  --visual-observations \
    outputs/tactical-sam3-11689-final-reviewed/visual-observations.json \
    outputs/tactical-sam3-11690-reviewed/visual-observations.json \
    outputs/tactical-sam3-11691-reviewed/visual-observations.json \
  --visual-video data/tactical/BAR-ATH_tactical.mp4 \
  --out outputs/collective-sam3-tactical-next
```

Resultado final: `outputs/collective-sam3-tactical-verified/`. O arquivo `visual-integration-check.json` confere que **somente 11689, 11690 e 11691 mudaram de X para H** frente à versão v3 com kickoff. O primeiro kickoff permanece A, com fonte de formação no centro. A referência pública foi consultada somente depois das previsões para métricas; não alimenta o classificador nem as observações de controle. São três casos dirigidos do mesmo recorte público já explorado, não avaliação independente ou validação de precisão.

O contrato mantém a disponibilidade declarada por frame. Revisões humanas aqui são offline e assumidas disponíveis no frame da imagem; isso não demonstra latência causal de um operador real. A animação é 25 fps, enquanto as inferências CPU são offline e levam dezenas de segundos por frame.

Nova animação: `outputs/simulacoes/tracking-gabarito-sam3-tactical-3frames/`, com legenda **“3 frames / bola nas mãos revisada”**. Still do caso: `outputs/tactical-sam3-integration-review/goleiro-tactical-antes-depois.png`. A cobertura visual não foi estendida aos 3.000 frames da janela, a bola automática não ficou confiável nesse caso e as referências de linha continuam pendentes. O progresso concreto é a integração verificável de SAM/contexto/controle revisado nos mesmos frames reais, sem inventar pixels de bola, corrigir tracking pela referência ou aprovar templates inválidos.

Validação desta continuação: **224 testes completos aprovados**, incluindo extração real de um vídeo sintético pelo contador correto, montagem completa do rascunho, projeção/ausência/atrasos, contatos bola/corpo, revisão vinculada aos assets e bloqueio de alteração de entradas durante a inferência. Os testes sintéticos não comprovam qualidade visual do SAM. O vídeo original e a prova real dirigida acima estão registrados separadamente.

A apresentação final para comparar o efeito visual usa **o mesmo coletivo antes/depois**, em `outputs/collective-sam3-before-after-comparison/` (`comparison_only_staging=true`), com origens/hashes dos dois resultados preservados. A animação em **`outputs/simulacoes/tracking-gabarito-sam3-antes-depois/`** mostra gabarito, “Sem visual” e “SAM 3 + revisão”. Não se recalculam previsões nessa montagem. O comparador histórico “Anterior” pode já dizer H nesse instante; ele não é o coletivo v3 sem as observações. O still correspondente é `outputs/tactical-sam3-integration-review/goleiro-tactical-comparacao.png`. Rótulos dos candidatos de encontro foram separados verticalmente no renderizador para evitar sobreposição no lance do goleiro; somente a renderização foi refeita.

## Evidências de jogo, continuidade e recuperação visual — v4

A política `collective-evidence-v4-experimental` acrescenta entradas opcionais a `CollectiveModel.feed`: `game_evidence` e `recovered_observations`. Os eventos do detector continuam usando exclusivamente as entradas originais e são idênticos ao comparador coletivo v3 com as três revisões SAM. Alterações locais anteriores foram preservadas.

### Bola morta e reinício

`src/team_possession/game_state.py` distingue evidências `interruption`, `restart` e `geometry`. Cada registro traz `evidence_id`, `frame`, `available_at_frame`, `valid_until_frame`, `source_id`, `reason`, verificação e revisor. Evidência atrasada não é reaplicada ao passado. A validade máxima admitida é dez segundos; as âncoras reais desta execução usam dois segundos, com renovação explícita. Conflito entre interrupção/reinício não escolhe um lado. Kickoff explícito mantém prioridade. D limpa origem, hipótese de passe e recepção; reinício ou expiração encerra D, sem inventar controle individual.

A regra de saída requer duas observações consecutivas válidas, origem dentro do campo, deslocamento de até 45 m/s e bola inteira além do limite (raio de 0,11 m mais incerteza geométrica revisada). Exige geometria verificada no frame; não aprova automaticamente homografia de solo para bola elevada. Bola parada, baixa velocidade e ausência não geram D. Não se copiam intervalos de Smart Tagging.

No caso real, a revisão da sequência original confirmou manipulação/colocação da bola para um reinício em 12915 e 12940; o chute observado em 12970 encerra D. Foram integrados **55 frames D, de 12915 a 12969**. Não se retroage ao início da ausência em 12831. O reinício termina D e deixa X até existir suporte de posse. Não há detector automático de apito/falta. O resultado é revisão offline por Codex, sem certificação humana nem demonstração de latência real.

### Continuidade de passe sem preenchimento retroativo

`pass_continuity.py` mantém recebedores observados que se aproximam em frames consecutivos, ou candidatos atuais sustentados por encontro/corredor de trajetória. Uma hipótese exige origem confirmada ainda dentro de quatro segundos, bola atual observada e válida, cobertura de movimento dos dois times, erros instantâneo/RMS até 0,75 m, movimento de passe e ausência de controle/contato adversário conflitante. Alterar o candidato não renova a idade do último controle. Perda de requisitos abandona os candidatos, com diagnóstico.

A saída inicial recebe hipótese separada `observed_sender_outgoing_launch`: bola a até 2 m de um passador observado, movimento para fora dele, adversário observado a mais de 2 m e todos os requisitos anteriores. A idade máxima é a tolerância já registrada de cinco frames / 25 Hz = 0,20 s; a hipótese não se transforma em passe permanente por ter começado. Os frames 11873–11874 são esse caso. Em 11915–11921, a aproximação do recebedor observado sustenta a continuidade apesar da falha instantânea do encontro previsto. Em 11513–11524, aquecimento de trajetória e oportunidade adversária preservam X. Nenhuma regra lê o rótulo posterior.

### Auditoria e recuperação visual pontual

`scripts/audit_tactical_observations.py` confronta os frames extraídos pelo contador original com ATD bruto, homografia, pontos projetados e elegibilidade de renderização. Nos 35 frames auditados, não se encontrou observação válida dentro do campo omitida pela regra do renderizador. A máscara de disponibilidade agora também evita desenhar pontos futuros. O deslocamento da bola projetada não foi corrigido por offset presumido; altura e erro de posição continuam causas possíveis.

`scripts/run_sam3_recovery.py` propõe pessoas na imagem inteira com SAM 3 baseline em CPU, uma execução por vez, watchdog de 60 s e RSS amostrado máximo de 7 GB. Não aprova candidatos. Foram executados os frames 11518 (24,97 s / 6,79 GB) e 12860 (25,80 s / 6,83 GB). Há propostas claramente inválidas e pessoas fora do campo; as demais propostas não foram promovidas automaticamente.

Duas observações foram revisadas no RGB, máscara e projeção:

- **11518:** pessoa de uniforme branco sem ponto ATD correspondente. Pés revisados em (1124,413) pixels, projeção ≈(48,51;26,47) m, distância de 4,05 m do ponto ATD observado mais próximo. Recuperação local A, sem identidade persistente ou interpolação.
- **12860:** o ponto bruto A26 coincide com uma pessoa de uniforme azul/vermelho; a projeção dos pés revisados fica a 0,28 m dele. Sobreposição local de equipe H, sem alterar o ATD, resolver identidade persistente ou estender a decisão aos frames vizinhos.

`recovery.py` verifica os hashes de vídeo, imagem, máscara, proposta SAM e homografia, o frame e a projeção inversa dos pés. Ausência de revisão, equipe incerta, pés ocultos, geometria indisponível, associação ambígua/duplicada ou observação atrasada impede integração. Correções exigem alvo observado disponível. As observações entram no conjunto de recebedores e na representação; não geram eventos do detector ou controle individual. A disponibilidade é declarada para replay offline, não medida em operação ao vivo. Os núcleos não viraram referências de classificação.

O renderizador usa marca magenta para observações/correções revisadas, mantém ausência de bola e apresenta D/X separadamente. O ponto A46 mal localizado junto ao goleiro H permanece diagnosticado; não foi removido ou reposicionado sem evidência suficiente.

### Execução e avaliação reproduzíveis

Pacote revisado final: `outputs/possession-v4-reviewed-evidence/evidence-final.json`; revisões e painéis em `outputs/possession-v4-recovery-review/` e `outputs/possession-v4-restart-dense-fixed/`. Cada saída nova deve ter destino novo.

```sh
.venv/bin/python -m scripts.run_collective_experiment \
  --data-dir data --baseline outputs/public-2min-missing-ball-v1 \
  --comparison-run outputs/collective-sam3-tactical-verified \
  --ground-truth evaluation/gt/BAR-ATH_possession_gt_public.csv \
  --visual-observations \
    outputs/tactical-sam3-11689-final-reviewed/visual-observations.json \
    outputs/tactical-sam3-11690-reviewed/visual-observations.json \
    outputs/tactical-sam3-11691-reviewed/visual-observations.json \
  --visual-video data/tactical/BAR-ATH_tactical.mp4 \
  --game-evidence outputs/possession-v4-reviewed-evidence/evidence-final.json \
  --recovered-observations outputs/possession-v4-reviewed-evidence/evidence-final.json \
  --out outputs/collective-evidence-v4-next

.venv/bin/python -m scripts.evaluate_evidence_windows \
  --selection outputs/possession-v4-selection/selection-final.json \
  --out outputs/possession-v4-reserved-next
```

O experimento congela código, configuração, arquivos de observação e assets referenciados antes da inferência, verifica a identidade novamente e salva previsões antes de abrir o gabarito público. `--comparison-run` usa o coletivo anterior salvo, não o consumidor histórico de eventos. Os modelos não recebem gabarito de posse.

No recorte público já explorado (11185–14184): acurácia **65,93% → 72,17%**, macro F1 **0,4467 → 0,6527**, X **772 → 534**, D **0 → 55** e lacunas curtas X entre a mesma equipe **33 → 18**. **O F1 oficial de segmentos piorou: 0,5091 → 0,4870.** Dos 238 frames alterados, 187 concordam e 51 discordam da referência. Não houve escolha de parâmetros pelo score; a hipótese inicial de saída de passe foi desenvolvida após a primeira avaliação exploratória e antes dos resultados reservados, ficando registrada como desenvolvimento.

As janelas posteriores congeladas (14500–15100 e 15400–16100) e os desafios de bola elevada/limite (17440–17480 e 18010–18150) totalizam 1.484 frames. Não houve continuidade nova sobre controle adversário observado. Em 17456 e 18025, as projeções além do campo foram recusadas como prova de saída devido à bola elevada/perto do goleiro. Sem gabarito independente de posse nesses trechos, não se reporta acurácia. Contatos visuais pontuais foram anotados antes da leitura das previsões reservadas e não convertidos em gabarito de posse. Reexecução após endurecimento de validações deve conservar os mesmos rótulos; não há ajuste de parâmetros pelos resultados reservados.

As saídas `*-first` e `*-validated` são intermediárias preservadas; a execução final é `outputs/collective-evidence-v4-release/`. O relatório final, a matriz de confusão, as transições e as limitações ficam em `outputs/possession-v4-assessment-release/`. A animação final fica em `outputs/simulacoes/tracking-gabarito-evidencias-v4-release/`, gerada com o helper da skill local a partir das previsões finais.

A integração segue experimental e dirigida: três controles anteriores de goleiro revisados, uma pessoa recuperada, uma correção local de equipe e três âncoras de interrupção/reinício. Não há reconhecimento automático completo de bola/equipe, reparo global do tracking ou referências de linha aprovadas. As 22 reprovações e a variante negativa 0,7 permanecem preservadas.

Verificação final: **240 testes completos aprovados**, incluindo disponibilidade de correções, vínculo à proposta SAM, reprojeção inversa, ausência de duplicação de identificadores entre ATD e sobreposição, e reinício revisado com controle atual concordante. Esse último caso pode restabelecer a equipe sem criar evento artificial; equipe conflitante ou bola indisponível preserva X. `git diff --check` sem erros. Os testes sintéticos não medem precisão visual real.

## Contexto causal de 25 frames e identificação do goleiro — v5

Pedido do usuário: testar um segundo de contexto a 25 Hz, sem abolir X, e corrigir a indicação do goleiro do Barcelona na simulação. Política: `collective-context25-v5-experimental`. O estado anterior continua preservado em `outputs/collective-evidence-v4-release/`.

### Histórico de evidências, não preenchimento de rótulos

`src/team_possession/temporal_context.py` mantém no máximo 25 registros, enfatizando os cinco mais recentes. O peso decai linearmente com a idade e é multiplicado por dois nesses cinco. Registros de controle observado têm força 1; evidências geométricas/visuais explícitas têm força 0,6. Uma decisão contextual **não é reinserida como evidência**: não renova a origem nem sustenta a si própria.

A continuidade contextual exige pelo menos cinco frames de evidência independente, suporte ponderado à origem ≥0,75, origem ainda dentro dos quatro segundos já existentes, bola atual observada/válida, cobertura dos dois times, trajetória não contraditória, adversário observado afastado, ausência de controle/toque adversário e de interceptação iminente. Movimento lento/aquecimento de trajetória só permite continuidade com controle nos cinco frames recentes e jogador da própria equipe observado a até 2 m. Não há inferência D por baixa velocidade. Ausência prolongada, D/reinício, quebra de continuidade e mudança de origem limpam a memória. X permanece diante de informação insuficiente.

Os parâmetros `context_frames=25` e `context_recent_frames=5` constam na configuração efetiva. Os snapshots trazem `temporal_context`, fração de suporte, quantidade de evidências, critérios, decisão base e dependências causais; `context_observed_frames` e `context_support_fraction` aparecem no catálogo de métricas. Qualidade contextual é uma categoria de evidência, não probabilidade calibrada.

### Causa do goleiro laranja e correção limitada

A conferência dos mesmos frames RGB/ATD/homografia encontrou mais que erro de cor. Em 11518, H24 coincide com o goleiro verde e A46 não tem pessoa correspondente; em 12400, A46 coincide com esse goleiro e H24 fica deslocado. Portanto, recolorir todo A46 como H criaria duplicações e classificaria incorretamente o goleiro visitante quando ele reaparece no extremo direito.

`scripts/prepare_goalkeeper_track_review.py` processa os 3.000 frames originais sequencialmente, sem reamostrar ou fabricar posições. Na região delimitada do gol H do primeiro tempo, um indício RGB de verde é associado apenas aos pontos observados H24/A46 existentes, com limite geométrico perto da base do componente. A equipe H depende do período/gol defendido confirmados pelo usuário e da revisão de fonte. O indício de cor, sozinho, não decide equipe nem aprova referências. O primeiro piloto aceitou um componente curto de tronco em 11873; esse caso foi reprovado no desenvolvimento. A variante congelada exige componente de pelo menos 20 pixels de altura. Não houve escolha pelo gabarito de posse nem varredura de limiares até melhorar score.

Pacote final: `outputs/context25-goalkeeper-association-reviewed/tracking-review.json`. Há **935 associações automáticas propostas** sobre contexto revisado: 543 selecionam o ponto H24 e 392 selecionam A46 como origem espacial do goleiro H. Há **2.065 abstenções de associação**; 935 propostas não significam 935 revisões humanas. Quando aceito, `reviewed-gk-H` usa exatamente a coordenada do ponto selecionado, com equipe H e origem registrada, evitando troca de equipe na identidade ATD original. Os pontos concorrentes são rejeitados como evidência, sem apagar o dado bruto. Em ausência de associação, não se inventa coordenada; o papel previamente revisado de H24 não certifica sua localização em cada frame.

O renderizador mostra G-H azul, G-A no goleiro visitante retomado à direita e cruz cinza nos pontos ATD rejeitados. Assim, o ponto A46 sobre/ao redor do goleiro H não continua apresentado como jogador confiável do Athletic. A correção só vale nesta janela revisada do primeiro tempo; não é uma regra geral “goleiro na esquerda = Barcelona”. Referências de linha reprovadas e revisão manual da bola nas mãos permanecem como antes. Não houve nova inferência SAM nesta etapa.

`tracking_review.py` aplica uma camada de qualidade e identificação sobre uma cópia do envelope. O ATD original não é modificado. O detector bruto é executado separadamente e preservado, enquanto o modelo corrigido usa as observações revisadas. Seus eventos podem diferir: 68 anteriores contra 65 corrigidos, com o fluxo bruto idêntico ao anterior. Essa mudança é registrada; não se afirma equivalência de eventos entre entradas agora diferentes.

### Teste separado das duas mudanças

Foram congeladas duas execuções antes da avaliação pública:

- Só contexto: `outputs/collective-context25-only-v5/`.
- Contexto + correção do goleiro: `outputs/collective-context25-goalkeeper-v5/`.

| Medida | Anterior v4 | Só contexto 25 | Contexto + goleiro |
|---|---:|---:|---:|
| X | 534 | 463 | 574 |
| Acurácia por frame | 72,17% | 74,30% | 73,27% |
| Macro F1 | 0,6527 | 0,6653 | 0,6778 |
| F1 oficial de segmentos | 0,4870 | 0,4870 | 0,5524 |
| D | 55 | 55 | 55 |

**O contexto isolado reduz X em 71 frames. A combinação não reduz o total de X frente à v4:** rejeitar pontos inconsistentes do goleiro também retira evidências antes usadas para preencher H/A, levando a 574 X. A combinação melhora acurácia e segmentos, mas esse efeito não deve ser apresentado como redução global dos desconhecidos. Não se afrouxaram critérios depois de consultar os scores. As lacunas curtas entre a mesma equipe passam de 18 para 16 no conjunto. O contexto foi aplicado em 76 frames nessa execução.

Os mesmos 1.484 frames posteriores foram usados como regressão, não como nova validação independente. Em `outputs/context25-regression-windows/`, há 57 decisões contextuais, nenhuma contra controle adversário observado e nenhuma janela excedendo 25 frames. Não se inventou acurácia fora da janela pública. O gabarito público continua sendo avaliação posterior e não entra nos módulos de decisão ou identificação visual.

### Reprodução e artefatos

O executor existente recebe agora `--tracking-review`, opcional. Para repetir, usar os mesmos parâmetros/observações da v4, trocar `--comparison-run` para `outputs/collective-evidence-v4-release` e acrescentar:

```sh
--tracking-review outputs/context25-goalkeeper-association-reviewed/tracking-review.json
```

Omitir a opção produz a ablação de contexto sem correção do goleiro. Configuração, entradas, imagens de evidência, política de lados/período, ATD e homografia são registrados por hash. A disponibilidade de revisões é declarada para replay offline, não mede latência de operação ao vivo.

A montagem `outputs/context25-three-model-comparison/` preserva hashes e origens das três previsões, sem recalcular os modelos. A simulação final está em `outputs/simulacoes/tracking-gabarito-contexto25-goleiro-verificado/`, com gabarito + anterior + contexto + contexto/goleiro, 3.000 frames, 25 fps, 120 s. O relatório está em `outputs/context25-goalkeeper-report/`, incluindo ablações, critérios de abstenção e diferenças por frame.

Verificação: **251 testes completos aprovados**. Os 11 novos testes cobrem expiração sem autoalimentação, ênfase recente, controle adversário, bola lenta, contexto insuficiente, equivalência por blocos, validação da janela, preservação do bruto, escopo da revisão, seleção sem duplicação, ponto futuro e abstenção do indício RGB. Os testes não certificam precisão visual. `git diff --check` sem erros.
