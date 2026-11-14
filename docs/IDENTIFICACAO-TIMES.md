# Identificação de times e goleiros

## Estado da entrega

Continuidade de controle por time e classificador de recortes implementados. As cores reais dos goleiros **não foram verificadas**: o vídeo não foi inspecionado. O link inicial retornou 404; um novo link autorizado permitiu obter o tracking real em uma execução posterior, mas não foram extraídos padrões visuais dos goleiros. Nenhuma cor real foi presumida. A conferência visual permanece pendente; os exemplos são sintéticos.

O detector recebe H/A resolvido. O time do ATD prevalece sobre cor; discordâncias ficam em `extensions.team_identification`. Sem metadado, a classificação visual estável pode fornecer o time. Árbitros e desconhecidos sem metadado são excluídos e retornam diagnóstico; o adaptador deve guardar e agregar esses diagnósticos. Não há detector de pessoas, ligação automática de vídeo/ATD nem consumidor final da frente 4 nesta entrega.

## API para recortes

`src.ball_events.uniforms.UniformClassifier(source, patterns, max_distance=.35, min_margin=.10)` recebe uma fonte e uma lista de padrões. Cada padrão contém `team` (H/A), `role` (`player` ou `goalkeeper`) e `crops` (lista de matrizes RGB, valores 0..255). Árbitros usam `role='referee'` e `team=None`. Goleiros devem ter padrões próprios associados ao clube. Vários padrões do mesmo papel/time são permitidos; sua distância mínima participa da comparação entre rótulos distintos.

As referências devem ter pelo menos 20 pixels de altura e 10 de largura. O tronco usa 20–55% da altura e 20–80% da largura. O descritor combina histograma HSV de tonalidade/saturação (24×8) e luminosidade (8), com pesos iguais e interpolação entre bins. A média dos descritores de cada conjunto forma um padrão. Camisas multicoloridas preservam a distribuição em vez de uma média RGB.

`classify(rgb, track_id, frame, available_at_frame=None)` compara todos os rótulos por distância de Hellinger, aceita distância até .35 e margem pelo menos .10. Valores são configuráveis, experimentais e não calibrados com a partida. A janela causal cobre os frames de `frame-4` a `frame`, exige três observações aceitas e 80% de concordância entre as aceitas. A classificação atual precisa concordar: imagem ambígua ou pequena retorna desconhecido imediatamente. Não usa frames futuros. O resultado contém papel, time, distância, margem, contagens, fonte, suporte e disponibilidade.

`resolve_player(player, visual=None)` retorna `(ponto_ou_None, diagnostico)`. Pontos resolvidos por cor incorporam o suporte e a disponibilidade visual; pontos com time de metadados mantêm sua causalidade original. O diagnóstico visual não modifica a decisão baseada no ATD. Reutilizar o resultado visual somente no seu próprio frame.

Instanciar um classificador por fonte e namespace de rastros; chamar `reset()` em cortes/descontinuidades, antes de alimentar o primeiro frame do novo segmento. Usar `forget(track_id)` ao retirar rastros para limitar memória. Não reaproveitar padrões entre tactical e broadcast sem validação.

## Continuidade e contrato de eventos

A persistência agora é por time. Cada frame ainda precisa de movimento individual verificável no próprio rastro; trocar o identificador em todo frame não cria controle. Distância, margem, duplicatas, saltos e qualidade continuam sendo verificados. A ausência interrompe a sequência candidata; não gera uma perda automaticamente.

Com bola observada, a hipótese ativa acompanha o time mesmo se o rastro original desaparecer. Perda exige afastamento persistente do jogador mais próximo desse time e cobertura válida dos dois times. Aquisição adversária substitui o estado ativo; nenhuma política final de posse é inferida.

Eventos continuam v2, mas consumidores precisam atualizar a interpretação: aquisição/perda aceitam um ou mais `track_ids` contribuintes; perda e aquisição precisam compartilhar time e segmento, não os mesmos rastros. Predecessor, suporte transitivo e disponibilidade continuam obrigatórios. Os motivos passaram a `same_team_close` e `sustained_team_separation_after_control_candidate`. Regenerar artefatos/cache com a nova identidade de código. A fixture causal foi regenerada para os novos motivos e hashes de eventos.

## Conferência obrigatória dos goleiros reais

Quando o vídeo estiver acessível, verificar ambos os goleiros em múltiplos momentos dos dois tempos; confirmar clube por contexto e metadados, sem deduzir apenas pelo lado do campo. Registrar fonte, frame/timestamp, descrição de cor, evidência do clube e eventuais trocas de uniforme. Cada goleiro precisa de padrão separado por fonte e de novas amostras se trocar de uniforme. Até concluir essa conferência, padrões reais permanecem sem configuração.

Guardar recortes, descritores ajustados e registro de inspeção em `outputs/uniforms/` (ignorado pelo Git), sem publicar assets ou derivados reais. Modelos sintéticos não representam uniformes da partida. Não usar gabarito de posse para escolher padrões ou limites.

## Demonstração e validação

```sh
.venv/bin/python -m scripts.demo_uniforms
.venv/bin/python -m unittest discover -s tests -v
```

A demonstração gera `outputs/demo_uniforms/report.json`, com classificação e aquisição de controle para ambos os goleiros sintéticos. Nenhuma imagem real é acessada. Os testes verificam ambiguidades, ambos os goleiros, árbitros, padrões repetidos, listras, brilho, prioridade ATD, continuidade por time e disponibilidade atrasada. Não demonstram precisão visual nem ganho esportivo em dados reais.
