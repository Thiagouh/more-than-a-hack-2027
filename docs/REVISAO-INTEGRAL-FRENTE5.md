# Revisão independente da frente 5

Data: 04/11/2026. Revisão solicitada pelo usuário e executada por subagente independente. Nenhum código de implementação, teste existente, asset ou avaliador oficial foi alterado. Este relatório é o único arquivo criado pela revisão.

## Parecer

A entrega possui uma base funcional coerente para experimentos sintéticos: os **32 testes passaram novamente**, a demo e o perfil de dois frames executaram, e os três hashes fixados do avaliador oficial conferem. A documentação distingue corretamente evidência heurística de posse final e reconhece a ausência de validação real.

Há, porém, uma falha de disponibilidade temporal que deve ser corrigida antes da integração: a perda pode chegar antes da aquisição da qual depende. Também foram reproduzidos problemas de cache, validação de entradas e auditoria. Os testes atuais não cobrem esses casos. A implementação está apta a continuar em desenvolvimento; não há evidência suficiente para declarar a integração temporal completa ou melhora de qualidade na partida.

Foram registrados **9 defeitos ou divergências reproduzidos**, **8 riscos ou limitações** e **9 melhorias adicionais**. P1 significa corrigir antes da integração; P2, corrigir antes de usar a funcionalidade afetada em experimento que produza conclusões; P3, melhoria de contrato, diagnóstico ou manutenção. As referências de linha correspondem ao estado revisado.

## Escopo e estado observado

- Repositório: `/Users/mac/Desktop/more than a hack/more-than-a-hack-2027`.
- HEAD observado: `954afaf` (`docs: introduce project to the community in English`).
- Alteração rastreada: acréscimo de `iter_aligned_points()` em `starter/load_atd.py`, sem mudança nas funções anteriores.
- Implementação nova, ainda não rastreada no Git: `src/ball_events/`, configurações, documentação e testes. Também havia ferramentas de avaliação e visualização não rastreadas, vindas do trabalho anterior; foram preservadas.
- Lidos integralmente: módulos de contrato, entrada, detector, artefatos, avaliação, CLI, demo, testes da frente 5 e os dois documentos indicados. Inspecionados também ferramentas compartilhadas, teste da avaliação/timeline, baseline, requisitos, regras de exclusão e pontos de integração.
- Não havia `AGENTS.md` no repositório ou na árvore do diretório de trabalho. O `AGENTS.md` do espelho do projeto ChatGPT foi lido: exige preservar as referências sincronizadas e não acrescenta regras de implementação.
- `data/tactical/` e `data/broadcast/` continham apenas guias. Não foi lido gabarito privado, não se executou avaliação pública e não se baixou tracking externo.

## A. Defeitos e divergências confirmados

### A1 — P1: perda omite a dependência temporal da aquisição

**Evidência:** `src/ball_events/detector.py:239` constrói a perda apenas com `self.loss`; `:258` guarda no estado ativo somente o jogador, descartando suporte e disponibilidade da aquisição.

**Reprodução:** bola em x=20 por oito frames, depois x=20,4;20,8;…;23,2; jogador H fixo em x=20 e A em x=26, a 25 fps. Declarando `available_at_frame=100` para o jogador A no frame 1, a aquisição sai com `(event_frame=1, available_at_frame=100)`, mas a perda sai com `(event_frame=13, available_at_frame=15, support_start_frame=10)`.

**Impacto:** um consumidor que respeite exatamente o contrato recebe a perda antes de poder conhecer a evidência de aquisição que autoriza sua existência. A ordenação final e `consume()` aceitam o resultado. Isso é uso prematuro de informação futura através de memória interna.

**Recomendação:** guardar no episódio ativo o ID e o resumo de dependências da aquisição; propagar suporte e disponibilidade para a perda. Explicitar a relação com `parent_event_id` ou equivalente. Testar aquisição atrasada por ponto e por pista, com separação ocorrendo antes de sua disponibilidade, e exigir que a perda não seja liberada antes da aquisição.

### A2 — P2: falha do cache opcional transforma execução concluída em erro

**Evidência:** `src/ball_events/artifacts.py:208` publica a saída e encerra o registro como concluído; `:209` inicia a gravação do cache fora desse bloco.

**Reprodução:** usar como `cache_dir` o caminho de um arquivo comum. A execução lança `FileExistsError`, mas a saída final existe e o registro de tentativa está `completed`.

**Impacto:** a CLI retorna falha após gerar todos os resultados; uma repetição no mesmo destino é recusada. Quem automatiza execuções recebe estados contraditórios.

**Recomendação:** validar o destino de cache antes de processar; tratar falha de cache como resultado separado e recuperável quando a saída está íntegra, ou incorporá-la explicitamente ao estado da tentativa. Cobrir permissão negada, caminho ocupado e falta de espaço por simulação.

### A3 — P2: parte da corrupção estrutural de cache não é recalculada

**Evidência:** `src/ball_events/artifacts.py:122` usa `artifact_sha256.items()` sem validar que o campo é um objeto; a captura em `:124` não trata `AttributeError`.

**Reprodução:** um manifesto com `cache_key` correto e `artifact_sha256` contendo uma lista com os seis nomes de `ARTIFACTS` passa na comparação de conjuntos e lança `AttributeError` em vez de invalidar o cache.

**Impacto:** contradiz a garantia documentada de recalcular conteúdo corrompido e bloqueia execução válida por causa de um cache descartável.

**Recomendação:** validar tipos e formato dos hashes antes de iterar; considerar qualquer estrutura inválida um cache miss. Acrescentar casos de manifesto com lista, null, campos ausentes e hashes de tipo incorreto.

### A4 — P2: referências JSON malformadas escapam do tratamento seguro da CLI

**Evidência:** `src/ball_events/evaluation.py:38` assume objeto na raiz e `:39` assume objeto em `metadata`. A CLI em `src/ball_events/__main__.py` não captura `AttributeError`.

**Reprodução:** executar `evaluate-events` com referência `[]`, ou `{"metadata":[],"events":[]}`. Ambos geram `AttributeError` sem o diagnóstico controlado. Há a mesma classe de problema em `inputs.py:44` se `files` for a lista `["players","ball"]`.

**Impacto:** falha abrupta em entradas inválidas comuns; a interface não mantém a promessa de diagnóstico uniforme. Não foi observado vazamento de registros nesses casos específicos.

**Recomendação:** validar objetos/listas e seus elementos em cada fronteira, levantando `InputError`. Não resolver apenas capturando indiscriminadamente qualquer exceção. Testar raiz, metadados, `files`, predições e itens de referência com tipos errados.

### A5 — P2: avaliação aceita duas cópias do mesmo evento como duas predições

**Evidência:** `src/ball_events/evaluation.py:55` seleciona toques; `:57` valida somente frame e time. A associação distingue índices de lista, sem conferir `event_id`.

**Reprodução:** duas cópias da predição H no frame 25, mesmo `event_id`, a 25 fps, contra duas referências H em 1,0 e 1,1 segundo. O resultado informa **2 correspondências** em 0,2 segundo, embora só exista um evento previsto distinto.

**Impacto:** concatenar acidentalmente arquivos ou repetir uma exportação pode inflar cobertura. O teste de associação um a um atual verifica índices distintos, não identidade de eventos.

**Recomendação:** exigir e validar IDs únicos para artefatos do detector, ou definir um modo simplificado explícito que gere/verifique identidades de entrada. Recusar duplicatas em vez de contabilizá-las duas vezes. Recusar também `event_frame=True`, atualmente aceito por ser subtipo de inteiro em Python.

### A6 — P2: o consumidor não valida o contrato completo que seu nome pode sugerir

**Evidência:** `src/ball_events/artifacts.py:67` verifica ordenação, unicidade e algumas relações temporais; não verifica versão, tipo, time, estrutura dos componentes ou tipos inteiros.

**Reprodução:** evento com `contract_version="v99"`, `type="invalid"`, `team="INVALID"` e campos temporais consistentes recebe `delivery_order_valid=True`.

**Impacto:** o arquivo `consumer_check.json` é insuficiente como aceite do contrato v1 entre frentes. Essa não é uma falha da decisão de posse, que ainda não está implementada; é uma lacuna na garantia de integração.

**Recomendação:** criar um validador compartilhado de evento v1 e chamá-lo na fronteira do consumidor. Receber metadados para conferir limites da timeline, permitir disponibilidade após o fim conforme documentado e verificar coerência dos campos de atraso. Até lá, rotular o resultado como conferência restrita de ordem e relações temporais.

### A7 — P3: `quality_flags` é obrigatório no CSV, mas opcional silenciosamente na API

**Evidência:** `src/ball_events/contract.py:113` usa `out.get('quality_flags', [])`; o guia apresenta o campo como obrigatório, e `inputs.py:56` exige a coluna no CSV.

**Reprodução:** remover `quality_flags` de um ponto enviado à API produz um ponto normalizado com lista vazia, sem erro.

**Impacto:** produtores Python podem deixar de declarar qualidade e passar a ter observações elegíveis para confirmação, enquanto o mesmo esquema em CSV seria recusado.

**Recomendação:** exigir o campo em todas as entradas ou documentar deliberadamente o default na API. Como flags bloqueiam confirmação, a opção conservadora é não transformar omissão em garantia de ausência de problemas.

### A8 — P3: o registro não cobre todas as tentativas alegadas pelo guia

**Evidência:** `src/ball_events/artifacts.py:155` valida configuração e `:157` calcula identidade antes de entrar em `attempt_log()` em `:168`; a CLI lê manifesto, pistas e configuração antes de chamar `run()`.

**Reprodução:** `run(..., overrides={"persistence_frames":1}, hypothesis="...")` é recusado sem criar registro de tentativa. Falhas de pré-processamento de manifesto também ficam fora do registro.

**Impacto:** o histórico não inclui tentativas recusadas por configuração/entrada, embora a documentação diga que cada tentativa iniciada fica registrada. Isso enfraquece auditoria experimental, sem invalidar por si só as saídas bem-sucedidas.

**Recomendação:** definir claramente o começo de uma tentativa e registrar desde essa fronteira, mesmo com identidade parcial; distinguir `rejected`, `failed_processing`, `completed` e erro de cache. Ou restringir expressamente a promessa do guia ao processamento iniciado após validação.

### A9 — P3: métrica chamada de latência mede erro de localização do evento

**Evidência:** `src/ball_events/evaluation.py:71` calcula `event_frame/fps - reference.time_s`; não usa confirmação ou disponibilidade.

**Reprodução conceitual direta da fórmula:** um toque corretamente localizado em 1,0 s, confirmado/disponível em 1,2 s, é reportado com latência 0,0 s, apesar de só poder ser consumido 0,2 s depois.

**Impacto:** comparar perfis de persistência ou produtores com lookahead usando essa métrica não mede o atraso que chega à frente 4. Valores negativos representam antecipação do instante estimado, não entrega antes da referência.

**Recomendação:** renomear para `event_time_error_s` e acrescentar atraso de confirmação e de disponibilidade, com médias/medianas/quantis e cobertura dos campos. Preservar sinal e definição de cada medida no relatório.

## B. Riscos e limitações metodológicas ou operacionais

### B1 — P2: associação gulosa pode perder correspondências possíveis

Em `evaluation.py:62`, a menor distância global é consumida primeiro. Caso reproduzido: referências H em 0,2 e 0,5 s; predições H em 0,30 e 0,02 s; tolerância 0,2 s. A função retorna 1/2, mas as associações 0,02→0,2 e 0,30→0,5 permitem 2/2. A estratégia está documentada, portanto não é divergência da implementação; é uma limitação do significado de `reference_recall`.

Recomendação: decidir antes do piloto entre preservar a métrica gulosa com nome explícito ou maximizar cardinalidade e minimizar erro entre soluções de mesma cardinalidade. Publicar a regra e pares escolhidos. Testar densidade de ações, empates e estabilidade da ordem dos arquivos. O custo atual constrói todas as combinações predição×referência; usar janelas temporais ordenadas para referências grandes.

### B2 — P1 para alegação de resultado: integração real com a frente 4 não foi medida

`consume()` não executa fila temporal, política de retroatividade ou decisão de posse. Não foi localizado outro consumidor funcional de `ball_events` além dos testes e do gerador de artefatos. A documentação reconhece isso corretamente. É uma etapa pendente, e não um resultado entregue.

Recomendação: teste ponta a ponta com o consumidor real, eventos fora da ordem de confirmação por causa dos atrasos, disponibilidade após último frame, reinícios de continuidade e ablação com/sem eventos nas mesmas condições. Definir explicitamente se o consumidor revisa frames passados.

### B3 — P1 para alegação de qualidade: dados reais e referência independente continuam ausentes

Os 32 testes demonstram cenários construídos, não qualidade no jogo. Os assets locais não estavam disponíveis; não foi possível verificar formato efetivo do ATD, cobertura da partida, orientação, distribuição de flags, frequência de saltos ou eventos reais. Tampouco foi executado o piloto IDSSE. A impossibilidade de distinguir bola aérea em XY é corretamente declarada.

Recomendação: manter essas limitações no aceite; executar primeiro conferência de integridade/cobertura local e depois o piloto previamente definido. Não transformar números de detecção, queda de X/D ou score público em comprovação isolada de melhora.

### B4 — P2: streaming do leitor não implica memória limitada de toda a execução

`Detector.history` usa três estados; candidato/perda são limitados pela persistência. Entretanto, `track_teams` cresce com IDs distintos; todas as pistas são carregadas e percorridas a cada evento; `run()` retém todos os eventos e trechos de origem, e o SVG retém toda a representação. `detect()` acumula também todas as features por escolha explícita de API.

Recomendação: documentar memória como O(IDs + pistas + eventos + mudanças de origem), além do histórico; medir partida completa e cenário com IDs novos/alternância de fonte a cada frame. Se necessário, exportar/ordenar eventos em disco e agregar a visualização. Não há evidência de estouro de memória no teste executado.

### B5 — P2: cache compartilhado não tem coordenação entre processos

`artifacts.py:211` remove entrada existente e `:216` renomeia uma nova. Outra execução pode ler/remover/publicar a mesma chave nesse intervalo. É um risco identificado por inspeção; não se executou um teste concorrente destrutivo.

Recomendação: publicação idempotente com exclusão mútua ou criação exclusiva; conferir novamente a entrada antes de substituir; nunca remover uma entrada válida só porque outro processo a publicou. Testar dois escritores e um leitor da mesma chave.

### B6 — P2: suporte temporal conservador inclui dados não usados

`detector.py:170` incorpora todos os pontos de até três estados anteriores em toda feature, mesmo quando a decisão depende só do frame atual. `event()` volta a incorporar features de cada estado de suporte. Isso pode atrasar um evento por jogador distante ou observação antiga irrelevante. É seguro contra uso precoce nesses caminhos, mas pode prejudicar utilidade temporal.

Recomendação: registrar dependências por característica/decisão e propagar apenas as necessárias, mantendo todos os pontos realmente usados na seleção e comparação. Fazer isso depois de corrigir A1, sem reduzir dependências reais para melhorar artificialmente atraso.

### B7 — P2: parâmetros expressos em frames mudam significado ao variar fps

O contrato aceita qualquer fps positivo, mas persistência de 3 frames significa 120 ms a 25 Hz e 300 ms a 10 Hz. O guia de dados externos alerta corretamente contra interpolação artificial, mas ainda é necessário decidir se os mesmos parâmetros em frames são apropriados ao novo conjunto.

Recomendação: definir se a hipótese é persistência em frames ou duração física; registrar conversão explícita quando mudar a frequência. Não generalizar o perfil de 25 Hz automaticamente ao SkillCorner de 10 Hz.

### B8 — P3: identidade de cache da API omite `hints_status`

`quality.json` inclui o argumento `hints_status`, mas `identity()` não o inclui. Nas rotas CLI usuais o manifesto/arquivo de pistas tende a diferenciar a chave; chamadas diretas da API com os mesmos inputs e pistas, mas outro status, podem reutilizar um resumo antigo. Constatação por inspeção, sem alegar ocorrência na demo.

Recomendação: incluir todos os argumentos que afetam artefatos determinísticos na identidade, ou derivar o status exclusivamente das entradas identificadas.

## C. Melhorias adicionais, sem defeito operacional comprovado

1. **P3 — Diagnóstico de qualidade mais completo.** `detector.py:267` conta flags globais derivadas e flags da bola; flags de produtores de jogadores não entram integralmente no resumo. Acrescentar cobertura por time/produtor, ausência de jogadores/adversários, flags de candidatos e motivos de inelegibilidade. Não confundir essas contagens com acerto.
2. **P3 — Metadados espaciais do ATD.** `starter/load_atd.py:145` segue a ordem de times/jogadores do baseline, o que é coerente com o código existente. Confirmar em fixture derivada apenas de esquema e no asset local nome dos times, dimensões, orientação, slots e ausência de namespaces XML inesperados. Não declarar erro de mapeamento sem o asset. Considerar diagnóstico para coordenadas implausíveis, sem recortar automaticamente bola fora do campo.
3. **P3 — Proveniência dos relatórios de avaliação.** `report-csv` e `evaluate-events` não registram hashes das entradas, versão da regra de matching/configuração ou identidade de código como `detect`. Adicionar esses campos e os pares associados para reproduzir conclusões. Separar as métricas de passes e chutes e reportar distribuição de erros além da média.
4. **P3 — Limites do validador numérico.** Recusar bool em todos os campos inteiros, validar tipo de objetos antes de `.get()` e proteger conversões de inteiros gigantes que podem gerar `OverflowError`. Exigir nomes/IDs não vazios também no adaptador ATD e regras consistentes para campos extras. São melhorias de robustez; não foi observado erro com entradas válidas nos testes executados.
5. **P3 — Configuração entre frentes.** Publicar esquema versionado de manifesto, ponto, evento e pista, mais fixture de referência da saída. Definir vocabulário de flags/tipos de pista e política de desempate. Documentar como reiniciar IDs de tracks e como representar recortes com origem temporal verificável, além do identificador textual.
6. **P3 — Visualização para uma partida completa.** A prévia existente foi inspecionada e é legível para a demo. A largura fixa de 1.280 px e marcas sobrepostas não suportam inspeção detalhada de 199.319 frames; oferecer recortes/zoom, indicação de disponibilidade além do fim, legenda de cores por fonte e por time e alternativa acessível a dicas por mouse. A demo não contém perda: acrescentar um cenário visual com esse evento, sem remover os testes específicos já existentes.
7. **P3 — Publicação reprodutível.** As mudanças novas ainda não estavam versionadas. Antes de publicar, revisar a lista completa com inclusão explícita de código/config/testes/docs e exclusão de outputs/assets/manifestos reais; criar um commit identificável. Não foi feito commit nesta revisão. Considerar CI com Python suportado e ambiente limpo usando os requisitos atuais.
8. **P3 — Fontes externas fixadas e auditáveis.** Guardar snapshot dos metadados de licença, versão, IDs/tamanhos e hashes dos XMLs pequenos antes do piloto; referências em branches mutáveis podem mudar. Nesta revisão, abriram-se somente páginas oficiais Metrica e SkillCorner para conferência geral; nenhum tracking foi baixado. A API Figshare não ficou acessível pela ferramenta de navegação, portanto os IDs, tamanhos e licença IDSSE documentados não foram revalidados nesta revisão. A conclusão sobre independência dos eventos/sincronização continua condicionada à inspeção do material efetivo.
9. **P3 — Organização e manutenção.** Centralizar validação de evento, extrair regras do detector em unidades nomeadas quando houver evolução e remover importações não utilizadas. Preservar determinismo e evitar refatoração ampla antes de corrigir A1. Acrescentar testes focados nos defeitos desta revisão e propriedades temporais, sem duplicar a suíte em testes que só espelham linhas de implementação.

## Verificações executadas e seus limites

| Verificação | Resultado | Alcance |
|---|---|---|
| `.venv/bin/python -m unittest discover -s tests -v` | **32/32 passaram**; execução da suíte reportou 0,235 s | Inclui os testes compartilhados de avaliação e timeline; todas as entradas da frente 5 são sintéticas |
| Hashes oficiais | Os três conferem | `possession_metrics.py`, `run_evaluation.py`, `validate_submission.py`, conforme constantes `OFFICIAL` |
| CLI `demo` | Retorno 0 | Gerou seis eventos: duas aquisições, três toques, uma disputa |
| CLI `detect` com `configs/frente5_dois_frames.json` | Retorno 0 | Aquisições confirmadas nos frames 2 e 21, contra 3 e 22 no default; isso mede comportamento sintético, não melhora |
| Escala sintética: 10.000 frames × 22 jogadores | 1,357 s; pico RSS do processo 27.230.208 bytes | Alimentação incremental de `Detector`; 3 estados de histórico e 22 IDs ao final. Não inclui parsing de ATD real, escrita de todas as saídas ou memória sob churn de IDs/eventos |
| Dependência aquisição→perda atrasada | Defeito A1 reproduzido | Dados integralmente sintéticos; aquisição disponível em 100 e perda em 15 |
| Cache apontando para arquivo comum | Defeito A2 reproduzido | Saída final presente, exceção e log `completed` |
| Manifesto de cache com estrutura errada | Defeito A3 reproduzido | `AttributeError`, em vez de cache miss |
| Referência JSON `[]` e `metadata=[]` | Defeito A4 reproduzido | Exceção sem tratamento na CLI |
| Predição duplicada com mesmo ID | Defeito A5 reproduzido | Duas referências contadas para um único evento distinto |
| Consumidor com versão/tipo/time inválidos | Defeito A6 reproduzido | Aceito quando relações temporais são consistentes |
| Ponto Python sem `quality_flags` | Divergência A7 reproduzida | Default silencioso para lista vazia |
| Configuração inválida com hipótese | Divergência A8 reproduzida | Nenhum novo registro de tentativa |
| Matching guloso em duas ações próximas | Limite B1 reproduzido | 1 associação reportada onde existem 2 possíveis |
| SVG/prévia | XML validado pela suíte e prévia PNG existente inspecionada | Legível na demo; não houve inspeção visual de uma partida completa |

Os ensaios adicionais usaram diretórios temporários removidos automaticamente. Não foram acrescentados testes permanentes nem feitas correções. As pequenas reproduções foram executadas diretamente pelas APIs e, quando indicado, pela função `main()` da CLI.

## Reprodução mínima do achado prioritário

Executar da raiz do repositório com `.venv/bin/python`:

```python
from src.ball_events import detect
from src.ball_events.demo import metadata, point

xs = [20.] * 8 + [20.4, 20.8, 21.2, 21.6, 22., 22.4, 22.8, 23.2]
frames = [
    dict(frame=f, ball=point(f, x), players=[
        point(f, 20., track_id='h', team='H'),
        point(f, 26., track_id='a', team='A'),
    ])
    for f, x in enumerate(xs)
]
frames[1]['players'][1]['available_at_frame'] = 100
_, events, _ = detect(metadata(len(frames)), frames)
for event in events:
    print(event['type'], event['event_frame'], event['available_at_frame'])
```

Resultado observado relevante: aquisição em 1 disponível em 100; perda em 13 disponível em 15. O toque intermediário é independente e não elimina a falha causal da perda.

## Ordem recomendada de trabalho

1. Corrigir A1 e criar teste que verifique dependências transitivas entre eventos, inclusive pistas atrasadas.
2. Corrigir A2–A6 antes de confiar no cache, na avaliação de referências e no aceite de integração.
3. Resolver A7–A9 e fixar definições de matching, métricas e auditoria antes do piloto.
4. Executar integração real com a frente 4 e validação local do ATD; manter separados aceite de contrato, desempenho computacional e qualidade esportiva.
5. Executar o piloto externo congelado, com proveniência e limitações; depois priorizar as melhorias de escala, visualização e manutenção pela necessidade medida.

## Fontes externas consultadas nesta revisão

- [Metrica Sports Sample Data — repositório oficial](https://github.com/metrica-sports/sample-data): conferência geral das descrições de coordenadas e dados de exemplo.
- [SkillCorner Open Data — repositório oficial](https://github.com/SkillCorner/opendata): conferência geral do catálogo e indicação de licença MIT.
- [API oficial Figshare](https://api.figshare.com/v2/articles/28196177): tentativa de acesso sem sucesso pela ferramenta; conteúdo não usado como nova evidência nesta revisão.

As fontes acima não foram usadas para validar empiricamente o detector. As recomendações e alertas metodológicos do documento externo original permanecem hipóteses/protocolo até o piloto ser executado.
