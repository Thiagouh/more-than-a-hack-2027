# Identificação visual de equipes — implementação por etapas

O fluxo novo é **imagem tática → máscara da pessoa → uniforme → pés/campo →
identidade local → associação ao ATD → observações corrigidas → posse e desenho**.
Não altera os arquivos ATD nem usa gabarito de posse para identificar equipes.
É experimental: implementação e testes não demonstram acurácia no jogo.

## Estado e entregas

| Etapa | Implementado | Liberação no jogo |
|---|---|---|
| Diagnóstico | Extração exata, projeções com slots, motivos de omissão e contagens | Diagnóstico inicial disponível em `outputs/team-identity-v1-review/` |
| Referências | Propostas de vários frames, SAM, painéis e congelamento existente | Exige revisão humana de uniformes, máscaras e ambos os goleiros; nenhuma aprovação automática |
| Associação | Pareamento global um a um, distância em metros e margem da alternativa global | Exige validação das correspondências em imagens revisadas |
| Continuidade | Identidades por movimento, memória causal de time e reinício por período | Parâmetros iniciais explícitos; ainda sem calibração/validação real |
| Integração | Camada única para modelo e desenho; incerteza neutra e exclusão da posse | Catálogo completo congelado obrigatório; não se mistura com correções visuais antigas |
| Avaliação | Equipe, cobertura, associações, perdas e trocas de identidade; comparação ATD | Anotações completas e congeladas antes da inferência de avaliação |

A inspeção inicial usa frame 11185 = **7min27,40s da gravação**, segundo o
alinhamento tático existente. Um atleta à direita não tem ponto projetado próximo;
não se resolve essa lacuna apenas trocando cores. A imagem e coordenadas aproximadas
do diagnóstico **não** constituem uma recuperação aprovada.

## 1. Preparar e revisar

Executar os comandos na raiz do projeto. Cada destino deve ser novo.

```bash
.venv/bin/python -m scripts.prepare_team_identity_review --out outputs/team-identity-review-novo
```

O comando prepara imagens originais, projeções do tracking, pranchas da entrada
em campo, início e trechos posteriores. A janela de desenvolvimento é 11185–14184;
a avaliação inicial contém sequências 16000–16004, 16250–16254 e 17000–17004.
São apenas três grupos exploratórios, sem alegação de validação da partida inteira.
Não se mudam esses trechos depois de consultar os resultados.

Em `annotations-draft.json`, completar **todas** as pessoas visíveis, incluindo
arbitragem, e marcar revisão, revisor e completude por frame. Cada pessoa usa os
campos do contrato visual v2: `label`, `box`, `role`, `classification`, `evaluable`,
`partial`, `lighting` e `occlusion`. Acrescentar `person_id` estável dentro da
sequência, `association_reviewed: true` e `expected_track_id` (string ou null
quando não há ponto correspondente). Equipes não são copiadas dos slots do ATD.

```bash
.venv/bin/python -m scripts.prepare_identity_truth --annotations outputs/team-identity-v1-review/annotations-draft.json --split development --out outputs/identity-dev-revisado --freeze
.venv/bin/python -m scripts.prepare_identity_truth --annotations outputs/team-identity-v1-review/annotations-draft.json --split evaluation --out outputs/identity-eval-revisado --freeze
```

Sem `--freeze`, o comando só gera rascunhos; imagens vazias ou não revisadas não
podem ser congeladas. A revisão de associação é independente da referência de posse.

## 2. Referências de uniforme

### Rodadas supervisionadas pelo usuário

`prepare_uniform_label_batch prepare` cria pranchas numeradas de recortes RGB,
sem rótulos sugeridos e sem depender de uma máscara SAM aprovada. Cada exemplo
guarda frame, caixa, hashes da imagem e do recorte e grupo temporal. A rodada
local `outputs/uniform-supervised-round2/` apresenta os exemplos 12–23, ainda
sem rótulos. O usuário pode responder por áudio ou texto usando os números.

As categorias são Barcelona, Athletic, Arbitro, Goleiro Barcelona, Goleiro
Athletic, Outro e Incerto. Registrar a resposta por `prepare_uniform_label_batch
record --batch ... --answers ... --out ...`, com o modelo `answers-template.json`.
O registro exige o lote inalterado, um revisor e uma resposta por número; não
aprova máscaras ou posições. O coletor aceita o período anterior ao jogo ou,
com `sampling_pool: development_gameplay` nas propostas, somente a janela
de desenvolvimento 11185–14184 com o jogo em andamento. As sequências de avaliação
permanecem excluídas. A quarta rodada declara o pool adicional `additional_gameplay`,
com a janela de desenvolvimento existente e intervalos adicionais exclusivos
de treino [25000, 36000) e [120000, 126000), incluindo
primeiro e segundo tempos. Esses intervalos foram escolhidos antes da rotulagem;
não podem ser usados depois como avaliação independente do classificador.
Rótulos aumentam a base
supervisionada; não representam treinamento concluído nem acurácia medida.

### Primeiro classificador supervisionado e teste separado

Depois da quinta rodada, `outputs/uniform-supervised-model-v1/model.json` foi
ajustado com 71 recortes confirmados (60 numerados e 11 iniciais): Barcelona 26,
Athletic 26, árbitros 9, goleiro H 4, goleiro A 5 e outro 1. O baseline usa o
descritor HSV do tronco já existente e o exemplo mais próximo por classe, com
distância de Hellinger. Limiares herdados .35/.10 foram congelados antes do teste.
É aprendizado supervisionado por exemplos; não é ajuste de uma rede neural.

`train_supervised_uniforms --batches ... --out ...` só aceita recortes revisados
com split `reference`, verifica hashes e rejeita duplicatas. O modelo guarda
rótulos, descritores, grupos e proveniência dos exemplos. Os recortes dispensam
aprovação das máscaras SAM para este baseline; isso não resolve a qualidade das
máscaras ou das posições da integração automática à posse.

O pool `held_out_test` contém as sequências reservadas 16000/16250/17000 e frames
adicionais 20000–20250, todos fora dos intervalos de treino. A sexta rodada,
IDs 72–87, tem split `test`. `test_supervised_uniforms predict --model ...
--batch ... --out ...` grava as previsões antes da resposta do usuário e exige
distância temporal mínima de 750 frames de qualquer exemplo de treino da fonte.
As previsões ficam em `outputs/uniform-supervised-model-v1/test-round6/`.

Após registrar as respostas pelo coletor, `test_supervised_uniforms score
--predictions ... --labeled ... --out ...` compara apenas as previsões salvas,
reportando acurácia, cobertura, abstenções, confusões por classe e quantidade de
grupos temporais. Não retreinar essa versão com os rótulos do teste. Esta primeira
prancha de teste não contém exemplo do goleiro A nem de equipe de filmagem;
portanto não valida essas categorias. É uma avaliação exploratória de recortes
manuais, não de detecção, tracking, partida inteira ou posse.

O primeiro teste resultou em 15/16 acertos, uma abstenção (ID 80, goleiro H) e
nenhuma classe aceita errada. Depois, por pedido explícito do usuário, uma cópia
dos 16 exemplos de teste foi promovida a treino para `uniform-supervised-model-v2`,
mantendo rótulos e avaliação originais. Essa versão tem 87 exemplos, sem mudar
algoritmo ou limiares. O campo `original_split` e o registro de promoção preservam
a origem dos exemplos. O teste anterior deixa de ser independente para v2.

A nova rodada usa frames 40000, 55000 e 70000, fora dos intervalos de treino e
distantes dos exemplos já incorporados. O executor também confere distância de
750 frames contra **todos** os exemplos efetivamente treinados, incluindo testes
promovidos, de modo que reutilizar a antiga prancha de teste em v2 é rejeitado.

O teste v2 resultou em 14/16 acertos e duas abstenções (92 e 95), sem classe aceita
errada. Por novo pedido explícito, os 16 rótulos foram copiados para treino de
`uniform-supervised-model-v3`, que tem 103 exemplos. O teste v2 permanece salvo
como resultado histórico e deixa de ser independente para v3. A rodada seguinte
usa frames 45000, 60000 e 80000, com IDs 104–119 e previsões comprometidas antes
dos novos rótulos; algoritmo e limiares continuam iguais.

### Quatro frames por atleta e casos difíceis

A interpretação supervisionada agora pode usar `FourFrameUniforms`, política
`supervised-four-consecutive-frames-v1`: janela móvel causal de **quatro frames
consecutivos da mesma identidade local**, equivalente a 0,12 s a 25 fps. Antes
do quarto frame a decisão é incerta. Lacunas, identidade nova ou associação
duvidosa reiniciam a janela; informação futura não entra no resultado atual.
São necessários pelo menos três frames com descritor válido e três apoios à
mesma classe. As distâncias por classe são combinadas pela mediana e passam
pelos limites .35/.10 existentes. Um frame isolado não muda a equipe; conflito
persistente continua incerto. Não foi medido ganho de acurácia desta política.

`run_team_identity --supervised-model outputs/uniform-supervised-model-v3/model.json`
seleciona esse fluxo no lugar de `--references`, mantendo os demais argumentos
de checkpoint, fonte e destino. A associação geométrica produz a identidade
antes da fusão de uniforme. O resultado de quatro frames alimenta diretamente a
camada corrigida, sem esperar adicionalmente os cinco votos da memória legada.
Árbitros e a classe Outro não viram candidatos de equipe. A fonte, o modelo e a
política ficam registrados no overlay consumido pela posse e pelo desenho.
O fluxo legado de referências/máscaras mantém seu comportamento anterior.

A rodada local `outputs/uniform-hard-sequences-round9-training/` contém **16
sequências, IDs 120–135, com 64 recortes reais**, divididas em duas pranchas.
Usa quatro grupos temporais já destinados a treino, incluindo corrida, perfil,
goleiros pequenos e atletas próximos. As caixas seguintes são propostas por
comparação RGB de curto alcance e tiveram continuidade visual conferida, sem
aprovação de pés, homografia ou máscaras. Nenhum rótulo de equipe foi inferido
para preencher as respostas do usuário.

`prepare_uniform_sequences prepare --proposals ... --extraction ... --out ...`
gera as pranchas e rascunhos. Uma resposta rotula a sequência inteira. Após
resposta do usuário, `prepare_uniform_sequences record --batch ... --answers ...
--out ...` só propaga o rótulo se a continuidade da pessoa nos quatro frames
estiver revisada. Exporta `labeled-batch.json` compatível com o treinador, com
IDs de sequência e grupos preservados. Não contar os quatro recortes como quatro
amostras independentes de avaliação nem distribuir uma sequência entre splits.
A prévia de inferência de quatro frames foi executada nas sequências reais e
salva separadamente; esta rodada é treino, não uma nova avaliação independente.

Identificar uma equipe não confirma bola em controle nem jogo em andamento.
Notas de paralisação ficam separadas das decisões de uniforme; esse classificador
não cria evidência de posse por reconhecer um goleiro.

### Teste da versão 4 com quatro frames

As 16 sequências de treino 120–135 foram rotuladas pelo usuário e exportadas em
`uniform-hard-sequences-round9-labeled`. No caso 127, prevalece a ressalva mais
específica: Athletic é a pessoa central, com Barcelona próximo. A versão 4 soma
167 recortes, incluindo os 64 recortes correlacionados dessas 16 sequências.

O teste seguinte, `uniform-four-frame-round10-test`, usa IDs 136–151 e sequências
50000–50003, 65000–65003 e 135000–135003, fora dos pools de treino e a pelo menos
750 frames dos exemplos efetivamente treinados. `prepare_uniform_sequences`
aceita `split: test` apenas nos intervalos de teste registrados e conserva esse
split ao exportar respostas, impedindo treino acidental pelos validadores atuais.

`test_uniform_sequences predict --model ... --batch ... --out ...` salva a
interpretação de quatro frames e o baseline do último frame sozinho, ambos com
o mesmo modelo/casos, antes de receber os rótulos. As previsões estão em
`outputs/uniform-supervised-model-v4-four-frames/test-round10/`. São 16 unidades
de avaliação e 64 imagens, não 64 observações independentes. A coleta ficou em
três grupos temporais e ainda é exploratória.

Após a resposta, registrar com `prepare_uniform_sequences record` e executar
`test_uniform_sequences score --predictions ... --labeled ... --out ...`.
O comando avalia uma resposta por sequência, exige rótulos posteriores às
previsões e produz `comparison.json` com as duas abordagens nos mesmos casos.
Nenhum desses novos rótulos entra no treino sem nova autorização explícita.

### Preparação e revisão de máscaras

`uniform-box-proposals.json` contém propostas de caixas, frame, time, papel e
iluminação. O formato é `tactical-uniform-box-proposals-v1`, com manifesto de
extração e seu hash. As propostas locais usam frames 7500 e 10500, anteriores ao
início do jogo, da própria câmera tática. Ainda precisam de revisão.

```bash
.venv-sam3-mac/bin/python -m scripts.prepare_identity_references --boxes outputs/team-identity-v1-review/uniform-box-proposals.json --checkpoint data/sam3-checkpoint/sam3.pt --checkpoint-sha256 9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e --out outputs/identity-candidatos-novo
.venv/bin/python -m scripts.review_team_references --manifest outputs/identity-candidatos-novo/candidate-references.json --out outputs/identity-revisao-nova
```

A escala de contexto é explícita (`--context-scale`, padrão 1.5); mudar o recorte
não aprova a máscara. Preservar variantes anteriores para comparação visual.
Nos painéis, inspecionar a imagem original, máscara e pixels usados no descritor.
Rejeitar halos, fundo, sobreposições e uniformes não verificáveis. Os campos de
aprovação só devem ser preenchidos depois dessa revisão. Os dois goleiros exigem
verificação própria; região do gol não substitui referência de uniforme.

Aplicar decisões pelo comando existente `review_team_references --decisions ...
--freeze-complete`, usando o modelo `review-draft.json`. O pacote completo exige
H/A de linha em sol/sombra, arbitragem e os dois goleiros. Catálogos reprovados
anteriores não são reutilizados. Não há transferência automática entre câmeras.

## 3. Executar identificação e avaliar

```bash
.venv-sam3-mac/bin/python -m scripts.run_team_identity --references outputs/identity-revisao-aprovada/frozen/frozen-references.json --checkpoint data/sam3-checkpoint/sam3.pt --checkpoint-sha256 9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e --start-frame 11160 --end-frame 14185 --out outputs/identity-dev-execucao
```

A inferência não recebe rótulos de avaliação. Usa o contador original do vídeo e
inclui 25 frames de aquecimento antes da comparação. Escreve `overlay.json`,
observações, máscaras recortadas e hashes de fonte, referências e código.
`--config` aceita somente os parâmetros declarados em `pipeline.DEFAULTS`.
Congelar a configuração no desenvolvimento antes da avaliação separada.

Para avaliar as sequências posteriores, executar uma nova inferência contínua
de 15975 até 17005, usando as mesmas referências e configuração. O conjunto
visual precisa ter sido congelado **antes** dessa execução.

```bash
.venv/bin/python -m scripts.evaluate_team_identity --truth outputs/identity-eval-revisado/frozen/frozen-truth.json --overlay outputs/identity-eval-execucao/overlay.json --baseline-atd data --out outputs/identity-eval-relatorio
```

O baseline ATD usa correspondências pessoa/slot revisadas: é uma comparação das
equipes com associação conhecida, não um benchmark independente de detecção.
`--baseline-overlay` permite comparar duas execuções visuais. O relatório separa
jogadores de linha e goleiros e não esconde perdas com precisão de aceitos.
Sem correspondências completas, a aprovação automática de expansão permanece falsa.

## 4. Posse e simulação

`run_collective_experiment` aceita `--team-identity caminho/overlay.json` e
`--visual-video data/tactical/BAR-ATH_tactical.mp4`, além dos argumentos já
existentes de baseline, dados, comparação, gabarito e destino. O overlay precisa
cobrir toda a janela; sua fonte e referências são verificadas antes do uso.
Não combinar com `--tracking-review`, `--recovered-observations` ou
`--visual-observations` antigos. Evidências de fase de jogo continuam separadas.

A entrada/saída da janela corrigida reinicia a memória do modelo para não carregar
controle atribuído a slots antigos. Fora dela, permanece o comportamento legado.
No trecho corrigido, somente equipes aceitas entram no detector; a animação lê
as mesmas observações salvas em `possession.states[].team_identity_players`.
Posições válidas sem time aparecem em cinza, com `?`; posições ausentes não são
inventadas. Ambiguidades de associação não geram jogadores duplicados.

Após comparar posse e identificação, gerar a simulação pelo helper local habitual.
Aprovar expansão apenas com menos erros de time, cobertura preservada, associações
e trocas de identidade sem piora, e regressão de posse aprovada separadamente.
Não publicar nem substituir automaticamente o modelo anterior.

## Verificação

```bash
.venv/bin/python -m unittest discover -s tests -q
```

Os testes novos exercitam ótimo global de associação contra busca exaustiva,
ambiguidade, geometria inválida, reaproveitamento de slots, memória, dados futuros,
neutralidade no desenho, compatibilidade legada e o fluxo máscara → equipe → posse.
As máscaras reais e as anotações humanas continuam sendo necessárias para avaliar
a qualidade no vídeo da partida.

### Comparação causal de janelas temporais — rodada 11

O experimento `scripts.compare_uniform_windows` preserva `FourFrameUniforms` e
sua política anterior. Compara offsets `[-3,-2,-1,0]`, `[-15,-10,-5,0]`,
`[-25,-21,-18,-14,-11,-7,-4,0]` e
`[-50,-45,-41,-36,-32,-27,-23,-18,-14,-9,-5,0]`, a 25 fps, no mesmo instante
final e com o modelo v4 congelado. A mediana das distâncias mantém os limiares
do modelo; descritores válidos e apoio à classe vencedora devem atingir
`ceil(.75 * amostras previstas)`, inclusive quando existem descritores ausentes.
A referência de quatro frames tem equivalência testada com a política anterior
para observações disponíveis sem lacunas. Nenhuma configuração está validada.

Cada caso conserva 51 observações reais, mesmo quando somente 4, 8 ou 12 entram
na classificação. O acompanhamento por pixels é apenas proposta. A revisão
visual de **todos** os frames intermediários fica em arquivo separado, ligado ao
hash do lote. Identidade ambígua, troca, frame ausente ou observação indisponível
fazem a configuração afetada abster-se. Não se usa slot ATD como identidade.
As pranchas `identity-review` permitem auditar esse acompanhamento; as pranchas
`rotular` mostram quatro instantes representativos sem previsões ou categorias.

A preparação confere a janela completa de dois segundos contra os pools de
treino, exemplos efetivamente presentes no modelo e frames encontrados nos
manifestos históricos de treino/teste (incluindo rótulos promovidos). Guarda o
inventário e hashes; margem mínima de 750 frames em relação aos exemplos.
Arquivos originais não são alterados. Novos trechos são escolhidos antes de
consultar previsões. Controles de integridade verificam pixels, modelo e revisão.

```bash
.venv/bin/python -m scripts.compare_uniform_windows prepare --model outputs/uniform-supervised-model-v4-four-frames/model.json --extraction EXTRACTION --proposals PROPOSALS --out NOVO_LOTE
.venv/bin/python -m scripts.compare_uniform_windows predict --model outputs/uniform-supervised-model-v4-four-frames/model.json --batch NOVO_LOTE/batch.json --review REVISAO_IDENTIDADE --out NOVAS_PREVISOES
.venv/bin/python -m scripts.compare_uniform_windows score --predictions NOVAS_PREVISOES/predictions.json --answers RESPOSTAS --out NOVA_AVALIACAO
```

Uma resposta rotula toda a sequência, sem exportação automática para treino.
O arquivo de respostas deve incluir `batch_sha256`, `reviewer`, `labels` (IDs
para uma das sete categorias existentes) e `reviewed_at_utc`, posterior ao
congelamento. `Incerto` como verdade humana não tem categoria verificável:
esses casos são contados separadamente e excluídos das taxas de acerto/cobertura.
O relatório mantém acertos, abstenções, erros, categorias, casos alterados e
quantidade de grupos. Casos do mesmo trecho e os frames de cada caso são
correlacionados; não equivalem a observações independentes da partida.

As janelas duram 0,12 / 0,6 / 1 / 2 s. Em regime contínuo, não esperam frames
futuros; no início de uma identidade, exigem esse histórico de aquecimento.
Latência computacional de produção não foi medida. Reconhecer uniforme,
inclusive em paralisação, não comprova posse ou jogo em andamento. Árbitro e
Outro continuam sem equipe candidata; nenhum rótulo deste teste entra no treino.

Entrega da rodada 11: `outputs/uniform-temporal-round11-ready/batch.json` contém
32 sequências (152–183) em três grupos (145000, 160000, 180000), com todos os
frames intermediários auditados visualmente. `identity-reviewed.json` registra
as decisões, sem rótulos de uniforme. Casos 153, 168 e 174 têm observações
omitidas a partir da primeira ambiguidade; propostas brutas permanecem somente
como diagnóstico separado. A etapa `compare_uniform_windows review` produz um
novo lote sanitizado: remove recorte e caixa das observações não verificadas,
sem substituir pixels nem recuperar artificialmente a identidade.

As quatro previsões foram congeladas em
`outputs/uniform-temporal-round11-predictions/predictions.json`; `integrity.json`
registra os hashes, a separação histórica e os 312 testes aprovados. As pranchas
ampliadas sem sugestões ficam em `outputs/uniform-temporal-round11-label-sheets/`,
produzidas por `scripts.render_uniform_window_labels` após o congelamento. O
modelo v4 e sua política anterior foram preservados; não houve retreino.
Não há rótulos novos nem resultado de acurácia desta comparação ainda.
Cobertura de sombra forte não foi confirmada. O caso 169 é um recorte parcial
de tronco, igual para as quatro políticas; isso limita comparações por categoria.

### Política escolhida para uso — quatro amostras em 0,6 segundo

Por decisão explícita do usuário após a rodada 11, o padrão supervisionado de
`IdentityPipeline` e `run_team_identity` passa a ser
`supervised-four-spaced-frames-0.6s-v1`, offsets **[-15, -10, -5, 0]** na timeline
original de 25 fps. Antes de 16 observações consecutivas da mesma identidade,
a saída continua incerta. Todos os 16 frames verificam continuidade; apenas
quatro descritores entram na mediana. Lacuna, identidade nova ou associação
duvidosa reiniciam o histórico. São exigidos três descritores válidos e três
apoios à classe vencedora; limiares do modelo permanecem inalterados.

O histórico guarda descritores, sem manter imagens RGB de frames passados.
Observações indisponíveis no instante da decisão não podem contribuir.
A política exige 0,6 s de aquecimento, sem esperar imagens futuras. O tempo de
processamento em produção não foi medido. Árbitros/Outro não viram equipes;
a política não altera evidências de posse ou de paralisação.

```bash
.venv/bin/python -m scripts.run_team_identity --supervised-model outputs/uniform-supervised-model-v4-four-frames/model.json --checkpoint data/sam3-checkpoint/sam3.pt --checkpoint-sha256 9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e --start-frame 11160 --end-frame 14185 --out outputs/identity-spaced-novo
```

Para reproduzir a política anterior, acrescentar
`--uniform-policy supervised-four-consecutive-frames-v1`. Os overlays registram
política, offsets, duração e tamanho do intervalo de continuidade. O leitor
aceita overlays históricos consecutivos, mas exige metadados completos e
coerentes para a política espaçada. O modelo v4, limiares e avaliações
congeladas permanecem intactos. Nenhum novo rótulo foi incorporado ao treino.

A avaliação final da rodada 11 exclui 173, 174 e 175 por solicitação explícita
do usuário; há 29 sequências avaliadas em três grupos temporais e nenhuma
pendência. Quatro frames em 0,6 s tiveram 26 acertos, três abstenções, zero erros
aceitos e cobertura de 89,7%. As exclusões ocorreram após a rotulagem; estes
resultados são exploratórios e não validam tracking automático na partida.
Os modelos, imagens e relatórios gerados continuam locais em `outputs/`, fora
do Git, conforme as regras do repositório.

### Desempenho — auditoria local após a seleção da política

A análise espacial agora calcula a sobreposição das máscaras uma vez por frame,
restringe interseções e erosão à região efetiva da pessoa e evita conversões e
validações redundantes de máscaras booleanas. Não altera limiares nem observações.
A equivalência foi conferida em 20 frames reais consecutivos, com saídas completas
idênticas, além dos testes de invariância de recorte/bordas e exclusão de valores
inválidos. A etapa CPU caiu de 2,025 para 0,561 s/frame (3,61x) neste Mac M1.

Isso **não** significa 3,61x na execução inteira: SAM completo ainda consumiu cerca
de nove segundos por frame em uma medição de dois frames. Não houve demonstração
de tempo real ou medição final de posse. A execução longa foi pausada para esta
auditoria; dados parciais permanecem locais em `outputs/`. O relatório está em
`outputs/identity-performance-audit-20261116/relatorio.md`.

Para processar 90 minutos, o caminho proposto é detecção leve + tracking causal
por frame + classificação nos quatro instantes reais, com SAM pontual e cache
verificado. Essa arquitetura ainda precisa ser implementada e validada quanto
a trocas de identidade e acurácia; não substituir silenciosamente o método atual.
