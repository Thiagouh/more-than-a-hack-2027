# Piloto de equipes com SAM 3

Classificação independente por aparição: H = Barcelona, A = Athletic Club, referee = árbitro e uncertain = incerto. Sem identidade persistente, números, ATD, contagem esperada ou gabarito de posse. Distância e margem são evidências, nunca probabilidades.

## Estado observado

A implementação local foi exercitada com testes sintéticos, adaptador SAM simulado e inferência real no Mac. O checkpoint oficial foi obtido após aprovação e autenticação do usuário; SHA256 verificado: 9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e. CPU concluiu o primeiro recorte com máscara não vazia; MPS concluiu uma tentativa sem candidatos e apresentou esperas prolongadas em outras tentativas, interrompidas explicitamente. **MPS ainda não é considerado estável.** Precisão e cobertura reais ainda não foram medidas; nenhuma integração à posse foi feita.

Os materiais reais permanecem em outputs/, ignorado pelo Git. O draft original foi preservado. A cópia outputs/team-pilot-reviewed-local/initial-reviewed.json contém revisão visual das 20 caixas de jogadores e duas de árbitros. A03/A04 foram ampliadas para incluir os pés; a nova imagem está em outputs/team-pilot-reviewed-local/box-review/initial-box-review.png. Há outros oficiais distantes/parciais fora dessas referências selecionadas. H09 está curvado e deve permanecer reprovado como referência enquanto seu tronco não for verificável. Sol foi identificado em H06/H07 e A05/A06; os demais candidatos foram marcados sombra. Confirmar a iluminação efetivamente coberta pelo tronco na revisão das máscaras.

A gravação de tela tem aproximadamente 32 s e cortes/saltos de reprodução. Os arquivos 16.png até 31.png são amostras posteriores; seus nomes e timestamps representam segundos da gravação, não frames nem tempo original da partida. Não há sincronização com o tracking.

## Contratos e proteção contra referências inválidas

- O descritor `masked-hsv-v3-core` usa os pixels RGB válidos do tronco: 20–55% da altura e 20–80% da largura da máscara original. Remove uma faixa interna de borda por erosão quadrada de raio `ceil(0,05 * largura)`, mínimo 1 pixel, com exterior da imagem inválido. A exclusão das sobreposições acontece depois, sem recalcular o enquadramento. Pacotes antigos precisam de nova revisão e congelamento; não são convertidos silenciosamente.
- Critérios provisórios: pelo menos 24 pixels, pelo menos 50% do tronco original restante, máscara com altura >=20 e largura >=10. Silhuetas com largura/altura >0,85 abstêm por postura/fusão suspeita.
- IoU >=0,85 sinaliza `duplicate_mask`. Caso contrário, interseção / área da menor máscara >=0,80 sinaliza `merged_or_contained_mask`. São regras **alternativas**, e ambas tornam as duas detecções incertas. Permanecem nos diagnósticos; não garantem detectar toda fusão ou todo tronco incorreto.
- Aprovação das caixas é separada de aprovação das máscaras. A geração produz apenas candidate-references.json: `mask_reviewed=false` e `reviewer=null` no manifesto inteiro, e `approved=false` em cada referência.
- O congelamento exige aprovação explícita, reviewer, árbitro, H/A de linha em sol e sombra, origem pre-kickoff dos jogadores de linha e goalkeeper_verified=true para cada referência de goleiro.
- Todos os candidatos, inclusive reprovados, participam da remoção de pixels compartilhados. Máscaras que escapam mais de 10% da caixa revisada são sinalizadas na geração e recusadas ao congelar e ao carregar o pacote. A máscara nunca é recortada para esconder esse vazamento.
- O pacote congelado copia imagens/máscaras para assets/, guarda hashes e descritores e pode ser transferido integralmente. A leitura verifica hashes e recalcula os descritores.
- Os limiares provisórios são distância máxima 0,35 e margem mínima 0,10. Não foram calibrados com os frames de avaliação. Caso seja necessário ajustá-los, separar um conjunto de desenvolvimento e reservar nova avaliação.
- Goleiros sem referência aprovada não têm template próprio. Aparências distantes/ambíguas abstêm; um classificador de cor isolado **não garante** distinguir um goleiro cujo uniforme se pareça com outra classe. Essa falha deve ser medida na avaliação; o piloto não está validado para goleiros.
- Entradas aceitam RGB e RGBA totalmente opaco, máscaras binárias 0/1/255 com dimensões exatas, caixas finitas dentro da imagem e rótulos seguros únicos.

O descritor concatena um histograma conjunto matiz/saturação (24 × 8 bins, peso 50%) e luminosidade (8 bins, peso 50%), com distribuição suave entre bins vizinhos. Usa distância de Hellinger: `||sqrt(d) - sqrt(ref)|| / sqrt(2)`. Para cada par `(equipe, papel)`, vale a menor distância às suas referências. A margem é a diferença entre o segundo e o primeiro par, não entre equipes agregadas. Linha H e goleiro H, por exemplo, competem; sem margem suficiente, a decisão é incerta mesmo quando ambos apontam H. Essa regra permanece inalterada nesta etapa. Distância/margem não são probabilidade.

## Comandos reproduzíveis

Executar na raiz do projeto. Usar uma pasta nova para cada saída; nenhuma etapa sobrescreve resultados existentes.

Tarefas leves verificadas no ambiente original `.venv/`: CPython 3.14.6, NumPy 2.5.4 e Pillow 13.0.0, fixados em `requirements-team-evaluation.lock.txt`. SAM no Mac usa o ambiente separado Python 3.12.14 e `requirements-sam3-mac.lock.txt`. Preservar ambos; não instalar dependências SAM no ambiente original. Cada saída registra as versões efetivamente usadas e hashes do código, mesmo com alterações ainda não commitadas. Os dois modos sintéticos foram exercitados sem PyTorch no ambiente leve.

Revisão local, sem modelo:

```sh
.venv/bin/python -m scripts.prepare_team_references \
  --manifest outputs/team-pilot-reviewed-local/initial-reviewed.json \
  --out outputs/team-pilot-box-review-next --boxes-only
```

No host externo, preparar um ambiente separado `.venv-sam3-cuda/`, instalar PyTorch/CUDA conforme o README oficial e instalar o SAM 3 a partir de checkout Git limpo no commit fixado. Os comandos CUDA abaixo pressupõem esse ambiente já preparado; ele não foi instalado nem testado neste Mac. Obter o checkpoint pelos meios autorizados; os executores não baixam pesos. Transferir o projeto, manifesto revisado e imagem, preservando os caminhos relativos. Não é necessário transferir o broadcast completo de 14 GB.

```sh
git clone https://github.com/facebookresearch/sam3.git /opt/sam3
git -C /opt/sam3 checkout --detach eef6839bcfc3458dd52ea80535dc6bd62aea86b1
.venv-sam3-cuda/bin/python -m pip install -e /opt/sam3
export SAM3_CHECKPOINT=/modelos/sam3.pt
export SAM3_SHA256=SUBSTITUIR_PELO_SHA256_AUTORIZADO_DE_64_HEXADECIMAIS

.venv-sam3-cuda/bin/python -m scripts.prepare_team_references \
  --device cuda \
  --manifest outputs/team-pilot-reviewed-local/initial-reviewed.json \
  --checkpoint "$SAM3_CHECKPOINT" --checkpoint-sha256 "$SAM3_SHA256" \
  --out outputs/team-pilot-gpu-candidates
```

A inicialização verifica o hash antes de importar o modelo e, no caminho CUDA, exige GPU CUDA, commit fixado e checkout sem alterações rastreadas. O adaptador usa caixas normalizadas cxcywh e máscaras booleanas conforme o código oficial local em data/sam3-source/sam3/model/sam3_image_processor.py. Para cada caixa, prefere alinhamento espacial e depois score; ausência de candidatos gera máscara vazia para revisão, nunca referência válida.

Inspecionar initial-mask-review.png e cada máscara candidata. Em candidate-references.json, preencher `reviewer` e `mask_reviewed=true` no topo e `approved=true` **somente** nas referências verificadas. Reprovar fusões, postura inadequada, pouco tronco e máscaras erradas. Não aprovar H09 automaticamente. Exemplo parcial de localização dos campos (os demais campos e hashes gerados devem ser preservados):

```json
{
  "mask_reviewed": true,
  "reviewer": "nome-do-revisor",
  "references": [
    {"label": "H02", "approved": true},
    {"label": "H09", "approved": false}
  ]
}
```

Para avaliação v2, cada referência também precisa de `source_sha256` (hash da gravação/fonte) e `source_segment_id` revisados. O gerador copia esses campos da anotação individual ou do topo do manifesto inicial; não inventa a origem. Pacotes históricos sem origem de trecho precisam de nova revisão e congelamento.

```sh
.venv/bin/python -m scripts.prepare_team_references \
  --manifest outputs/team-pilot-gpu-candidates/candidate-references.json \
  --freeze --out outputs/team-pilot-frozen

.venv/bin/python -m scripts.prepare_team_evaluation \
  --frames-dir outputs/team-pilot-initial/screen-recording-frames \
  --recording '/dados/Gravação de Tela 2026-11-14 às 21.12.20.mov' \
  --mode full_frame --sample-kind main \
  --out outputs/team-pilot-evaluation-draft-v2
```

Preparar o rascunho de avaliação **não exige referências congeladas**. Os segundos padrão 16,19,22,25,28,31 são apenas amostras da gravação, sem independência presumida. O rascunho deixa trechos, grupos e região vazios para revisão. As imagens mantêm hashes do arquivo e dos pixels RGB, dimensões e origem; registrar como foram extraídas (comando/ferramenta, escala, compressão e cortes) em `preparation`.

## Avaliação visual independente

### População e anotação v2

Revisar antes de gerar ou consultar decisões classificadas. Definir em cada imagem `evaluation_region`: lista de polígonos, cada um como `[[x,y], ...]`, cobrindo campo e áreas de atuação da arbitragem. A inclusão usa o **centro da caixa**, com borda incluída. Anotar todas as pessoas dentro dessa união, inclusive atletas, oficiais, parciais e pessoas fora do campo que estejam na região. Comissão, gandulas e espectadores dentro da região recebem `role=other`; público fora dela não entra na avaliação principal. Pessoas de papel desconhecido recebem `unknown`.

Cada pessoa exige `label` local, `box` xyxy dentro da imagem, `role` (`player/goalkeeper/referee/other/unknown`), `classification` (`H/A/referee/uncertain`), `evaluable`, `partial`, `lighting` (`sun/shade/mixed`) e `occlusion` (`isolated/overlapped`). `other/unknown` exigem `classification=uncertain` e `evaluable=false`. Atleta/árbitro cujo uniforme não possa ser decidido também recebe classificação incerta e não avaliável. Não atribuir papel ausente a partir da equipe.

Exemplo de pessoa parcial, com goleiro visualmente avaliável:

```json
{
  "label": "p01", "box": [10, 20, 40, 100],
  "role": "goalkeeper", "classification": "H",
  "evaluable": true, "partial": true,
  "lighting": "shade", "occlusion": "overlapped"
}
```

Preencher `source_segment_id` com o trecho contínuo de origem e `group_id` com o lance/grupo correlacionado, agrupando também seus replays. Reutilizar o mesmo identificador de grupo para o mesmo lance dentro da fonte, mesmo após um corte. `temporal_origin_known=true` exige `source_time_s` da partida/fonte original; desconhecimento exige `false` e `null`, sem converter segundos da gravação em segundos da partida.

Registrar trechos de referências e desenvolvimento em `excluded_segments`, por exemplo `{"source_sha256":"HASH_REAL_DE_64_HEXADECIMAIS","source_segment_id":"pre-kickoff","split":"reference"}`. `split` aceita `reference` ou `development`. O trecho de avaliação não pode coincidir com os registrados; todas as referências selecionadas precisam constar no registro. O código também impede reutilizar pixels RGB das referências. A revisão humana deve identificar replays e trechos equivalentes entre arquivos diferentes: nomes e hashes sozinhos não provam independência.

Após revisão completa, preencher `reviewer`, `reviewed=true` e `annotations_complete=true`. Congelar antes de produzir resultados:

```sh
.venv/bin/python -m scripts.prepare_team_evaluation \
  --freeze outputs/team-pilot-evaluation-draft-v2/visual-truth-draft.json \
  --out outputs/team-pilot-truth-v2

.venv/bin/python -m scripts.prepare_team_evaluation \
  --truth outputs/team-pilot-truth-v2/frozen-truth.json \
  --references outputs/team-pilot-frozen/frozen-references.json \
  --out outputs/team-pilot-run-v2

.venv-sam3-mac/bin/python -m scripts.run_team_pilot \
  --manifest outputs/team-pilot-run-v2/run-manifest.json \
  --device cpu \
  --checkpoint data/sam3-checkpoint/sam3.pt \
  --checkpoint-sha256 9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e \
  --out outputs/team-pilot-results-v2

.venv/bin/python -m scripts.evaluate_team_pilot \
  --truth outputs/team-pilot-truth-v2/frozen-truth.json \
  --detections outputs/team-pilot-results-v2/detections.jsonl \
  --out outputs/team-pilot-evaluation-report
```

O congelamento copia imagens e, quando aplicável, máscaras para `assets/`. O executor valida o pacote e registra seu hash em `provenance.json` **antes** de gerar decisões. Só RGB e máscaras chegam ao classificador. Saídas: frames/, masks/, detections.jsonl com motivos, contagens internas e vínculos por hash; provenance.json com modo, versão do protocolo, ambiente, revisão e hashes do código. O avaliador verifica imagem, pacote, modo e hash do arquivo de detecções; execução incompleta é recusada. Alterar gabarito ou imagens exige novo pacote e nova execução. Esse compromisso não comprova que o revisor nunca viu decisões anteriores.

### Dois modos, relatórios separados

- `full_frame`: SAM na imagem inteira, seguido de classificação. Associação de caixas um a um, independente da classe, por cardinalidade máxima com IoU >=0,5 e preferência determinística local por maior IoU. Detecções extras entram na precisão; extras com associação possível são duplicatas. Pessoas sem associação são perdas. Detecções cujo centro fique fora da região são apenas `outside_region_detections`. Decisões aceitas sobre pessoas `other/unknown` dentro da região são erros.
- `verified_regions`: imagem e máscaras individuais verificadas, sem executar SAM. Preparar outro rascunho com `--mode verified_regions`; cada pessoa exige `mask` (caminho relativo ao rascunho ou absoluto), `mask_sha256` e `mask_reviewed=true`. Aqui esse campo pertence à **pessoa anotada**, diferentemente do manifesto de referências. Máscaras binárias devem ter as dimensões da imagem, ser não vazias e caber na caixa anotada. Congelar, criar o run-manifest e executar com `.venv/bin/python -m scripts.run_team_pilot --manifest CAMINHO --out NOVA_PASTA`, sem checkpoint. Associação pelo `region_label` congelado, sem olhar classe; todas as regiões devem estar presentes uma vez. Máscaras compartilhadas continuam excluídas. Esse resultado mede classificação condicionada a segmentação verificada, não detecção, segmentação automática nem qualidade de imagem inteira.

Os modos nunca são somados em um relatório. No modo verificado, a inclusão na região é decidida pela caixa da anotação congelada; no fluxo completo, pela caixa prevista.

### Métricas e suficiência exploratória

`precision_accepted` = decisões aceitas com equipe/classe correta / todas as decisões aceitas. `precision_team_and_role` exige também papel correto, com o mesmo denominador. Um goleiro H previsto como jogador de linha H acerta equipe, mas erra o critério conjunto. `coverage` = atletas avaliáveis associados a decisão aceita / todos os atletas avaliáveis, incluindo perdas e goleiros. Cobertura mede aceitação, não acerto; decisões erradas são penalizadas na precisão. Árbitros participam da precisão e não da cobertura de atletas. Anotações incertas nunca tornam uma decisão aceita correta.

O relatório apresenta numeradores/denominadores, perdas, incertos, extras, duplicatas, matriz de confusão de papéis (incluindo `unknown` e `missed`) e estratos de linha, goleiros H/A, árbitros, iluminação e sobreposição, com cruzamentos por papel. Estratos de papel abrangem apenas pessoas anotadas/associadas; extras sem associação permanecem na precisão global e não recebem papel verdadeiro inventado. Papéis sem exemplos têm contagens zero e métricas nulas.

As metas pontuais continuam precisão de equipe/classe >=95% e cobertura >=80%, separadas do acerto conjunto. `point_targets_observed` (também exposto pelo nome histórico `targets_observed`) não significa validação. `sampling.sufficient_for_exploratory_analysis` exige: conjunto `main`, pelo menos 30 grupos independentes, 300 decisões aceitas, 30 aparições avaliáveis de goleiros (>=10 H e >=10 A), atletas avaliáveis em sol, sombra, isolados e sobrepostos, origem temporal conhecida e evidência não sintética. São mínimos operacionais, não garantia de poder estatístico.

O conjunto principal permite um frame por grupo; sequências do mesmo lance ficam em pacote separado `--sample-kind challenge`. Frames com pixels idênticos são recusados. Referências, desenvolvimento e avaliação usam trechos separados; qualquer mudança de limiar/descritor após consultar avaliação exige reservar nova avaliação. Os seis frames atuais continuam exploratórios e não satisfazem os mínimos presumidamente.

Intervalos de 95% usam bootstrap percentil por grupo (2.000 repetições, semente 0), preservando correlação entre pessoas/frames do grupo. Menos de 10 grupos ou origem temporal desconhecida omitem os intervalos. Cada métrica também exige denominador não zero em pelo menos 10 grupos; estratos preservam grupos sem exemplos como contribuição zero. Sem denominador, o intervalo é nulo. Bootstrap com zero erros pode produzir intervalo degenerado: não interpretar como certeza de precisão perfeita. Nenhum relatório autoriza automaticamente integração à posse; gabarito de posse e tags ATD não entram no protocolo.

### Compatibilidade e exemplos reproduzíveis

Rascunhos, resultados e manifestos de execução antigos não entram automaticamente no avaliador v2. Reanotar papéis, região, grupos e origem; criar novo pacote congelado e nova execução. Referências do descritor atual podem ser reaproveitadas somente após revisão explícita dos metadados de origem e novo congelamento; nenhuma referência real foi aprovada por esta mudança.

Exercitar ambos os fluxos com pixels sintéticos e SAM simulado, sem carregar pesos:

```sh
.venv/bin/python -m scripts.demo_team_evaluation --out outputs/team-evaluation-v2-synthetic-new
```

Cada modo gera entradas, referências sintéticas, gabarito congelado, resultados e relatório separados, com `synthetic=true` e `model_executed=false`. Os relatórios não medem qualidade visual real. Proveniência de preparação, congelamento, execução e avaliação registra Python, versões instaladas de NumPy/Pillow/PyTorch/torchvision, revisão Git e árvore alterada; a execução registra hashes dos arquivos de código. Não depende de ativação de shell nem do comando global `python`.

Verificação da implementação v2: **167 testes da suíte completa passaram** no ambiente original. Os exemplos completos foram executados em `outputs/team-evaluation-v2-synthetic-final/`, com relatórios separados em `full_frame/report/evaluation.json` e `verified_regions/report/evaluation.json`. Ambos declaram evidência sintética, ausência de execução de modelo e amostra insuficiente. Nenhuma nova inferência real foi executada nesta etapa. Referências reais e métricas reais de equipe/goleiros continuam pendentes.

## Testes locais

```sh
.venv/bin/python -m unittest tests.test_team_pilot tests.test_team_pilot_safety tests.test_team_evaluation_v2 tests.test_sam3_mac
.venv/bin/python -m unittest discover -s tests
```

Os testes de fluxo usam SAM simulado e imagens sintéticas, não medem qualidade do SAM ou dos uniformes reais.

## Mac Apple Silicon: caminho experimental de imagens

Instalado neste Mac M1 de 16 GB: Python 3.12.14, PyTorch 2.10.0 e torchvision 0.25.0 em .venv-sam3-mac/. O ambiente original .venv/ foi preservado. O checkout oficial fica em data/sam3-source/, no mesmo commit fixado e sem alterações. Ambos ficam fora do Git. As dependências testadas estão fixadas em requirements-sam3-mac.lock.txt.

O adaptador mac_compat.py modifica em memória sete módulos do commit verificado: dispositivos dos caches, transferência do modelo, RoPE real no backbone, divisão das frequências complexas do checkpoint em componentes reais/imaginárias, MLP com operações PyTorch usuais e tratamento de prompts vazios. A verificação de padding usa CPU explicitamente para evitar operações de validação problemáticas em MPS; a chamada CUDA de memória fixada foi removida desse caminho. A dependência EDT de vídeo é desativada com erro explícito se chamada; não substitui nenhuma máscara. Só se aplica com --device mps/cpu, nunca ao caminho CUDA. Não implementa tracking por vídeo. A proveniência de inferência registra a versão e o hash do adaptador. Não há migração silenciosa da inferência para CPU nem redução das proteções de memória do MPS.

Para recriar em outro Mac, na raiz do projeto (usar pastas novas se já existirem):

```sh
uv venv --python 3.12 .venv-sam3-mac
uv pip install --python .venv-sam3-mac/bin/python -r requirements-sam3-mac.lock.txt
git clone https://github.com/facebookresearch/sam3.git data/sam3-source
git -C data/sam3-source checkout --detach eef6839bcfc3458dd52ea80535dc6bd62aea86b1
uv pip install --python .venv-sam3-mac/bin/python --no-deps -e data/sam3-source
uv pip check --python .venv-sam3-mac/bin/python
.venv-sam3-mac/bin/python -m scripts.probe_sam3_mac --out outputs/mac-probe-new
```

O probe sem checkpoint testa componentes, sem construir o modelo completo ou baixar pesos. A saída declara segmentation_executed=false. Foi confirmado MPS disponível, cálculo de MLP, cache posicional, construção do decoder e comparação de atenção rotatória real com a referência complexa em CPU. Os números de tempo/memória desse teste **não representam** a inferência do modelo completo.

Quando houver checkpoint autorizado e SHA256, medir **uma caixa** primeiro:

```sh
.venv-sam3-mac/bin/python -m scripts.probe_sam3_mac \
  --checkpoint /caminho/autorizado/sam3.pt --checkpoint-sha256 HASH_AUTORIZADO \
  --manifest outputs/team-pilot-reviewed-local/initial-reviewed.json \
  --out outputs/mac-single-box
```

Esse comando mede carga, inferência, memória do processo e memória MPS; salva single-box-mask.png para inspeção. No estado atual, usar CPU explícita e apenas uma caixa antes de ampliar. MPS permanece experimental e instável. `prepare_team_references` mantém padrão CUDA; o executor v2 `run_team_pilot` usa CPU por padrão. Sempre informar `--device` nos experimentos reais para evitar ambiguidade.

### Primeiros resultados reais no M1

Pesos em data/sam3-checkpoint/sam3.pt, com origem/revisão e verificação registradas em source.json, fora do Git. A autenticação permanece no cache privado do Hugging Face, sem tokens nos arquivos do projeto.

- outputs/team-pilot-mac-single-box-diagnostic/mac-probe.json: MPS, carga 13,0 s, inferência 13,8 s, zero candidatos aceitos na primeira caixa; driver MPS alocou cerca de 8,0 GB. Não é uma máscara válida.
- outputs/team-pilot-mac-single-box-cpu-crop/mac-probe.json: CPU, carga 12,8 s, inferência 11,7 s, 1.898 pixels de máscara, score SAM 0,884, pico RSS de aproximadamente 6,4 GB. O campo device=cpu é a identificação correta do backend; o nome do teste de aritmética nesse primeiro relatório ainda dizia MPS por um erro de texto posteriormente corrigido.
- outputs/team-pilot-cpu-mask-review/H01-review.png: inspeção ampliada da máscara real da CPU. A máscara inclui fundo excessivo ao redor do jogador; H01 não foi aprovado como referência.

Essas observações não são benchmark de precisão nem garantia de desempenho. A CPU foi escolhida explicitamente para gerar os candidatos iniciais devido à instabilidade MPS.

```sh
.venv-sam3-mac/bin/python -m scripts.prepare_team_references \
  --device cpu --box-context-crop \
  --manifest outputs/team-pilot-reviewed-local/initial-reviewed.json \
  --checkpoint data/sam3-checkpoint/sam3.pt \
  --checkpoint-sha256 9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e \
  --out outputs/team-pilot-cpu-reference-candidates-next
```

--box-context-crop usa um recorte quadrado de lado 3 vezes a maior dimensão da caixa, limitado à imagem. O SAM opera no recorte, e a máscara retorna às coordenadas e dimensões originais. As cores do descritor continuam vindo dos pixels originais, sem interpolação. O prompt person precede a caixa. Nenhum limiar foi reduzido para obter máscaras. Os candidatos continuam sem aprovação automática; inspecionar initial-mask-review.png antes de congelar qualquer referência.

A execução CPU das 22 caixas terminou: outputs/team-pilot-cpu-reference-candidates/ contém máscaras reais, initial-mask-review.png e candidate-references.json. As 22 máscaras são não vazias, mas isso não implica qualidade suficiente. A revisão ampliada está em outputs/team-pilot-cpu-mask-contact-review/reference-contact-sheet.png e as observações em review-findings.json. Foram encontrados halos de fundo (por exemplo H01/H07/H08/H10/R01) e regiões parciais em pessoas sobrepostas. Nenhuma referência foi aprovada/congelada. Os critérios geométricos automáticos atuais não detectaram esses problemas de contorno, portanto quality_reason=null não constitui aprovação. A próxima etapa é melhorar e verificar a máscara do tronco, mantendo referências de H/A em sol/sombra; nenhuma meta de classificação foi medida.

Validação após as adaptações reais: 24 testes direcionados no ambiente Mac e 142 testes da suíte completa passaram. O checkout oficial permanece limpo no commit fixado; checkpoint, imagens e resultados continuam ignorados pelo Git.

### Auditoria de tronco sem modelo

```sh
.venv/bin/python -m scripts.audit_team_masks \
  --manifest outputs/team-pilot-cpu-reference-candidates/candidate-references.json \
  --out outputs/team-pilot-torso-core-audit-new
```

Verifica hashes dos candidatos, mantém pixels compartilhados excluídos e salva folhas com **somente os pixels do tronco usados no descritor**, além de métricas e motivos em audit.json. Não executa o modelo, altera candidatos ou concede aprovação.

A primeira auditoria v3 está em outputs/team-pilot-torso-core-audit-v3/. A inspeção confirma que a erosão não resolve halos largos: H01/H08 ainda incluem fundo no tronco; H09 continua inadequado pela postura. As 22 máscaras passaram nas heurísticas geométricas, portanto `quality_reason=null` continua sem significado de aprovação. Não há evidência de um deslocamento simples na restauração de coordenadas; a causa da imprecisão do modelo permanece em investigação. Nenhuma referência foi congelada e nenhuma meta de classificação foi medida.

Foi realizado também um teste real isolado H01 em CPU, mantendo recorte contextual e limiares, mas omitindo `set_text_prompt('person')`: `add_geometric_prompt` usa então o prompt oficial `visual`. Saída em outputs/team-pilot-cpu-geometric-only-H01/. Carga 15,97 s, inferência 12,20 s, pico RSS 6.210.420.736 bytes, 1.272 pixels, score 0,891. A máscara menor ainda contém halo visível: hipótese insuficiente, nenhuma aprovação e nenhuma execução ampliada. Esse experimento não altera o comportamento padrão do piloto. O resultado não demonstra que outra fonte de pixels ou outra estratégia de segmentação resolveria o problema; isso ainda precisa ser testado.

Validação desta etapa: 145 testes completos passaram no ambiente original, incluindo erosão sem reenquadramento, exclusão de sobreposições, vazamento de caixa sem recorte e auditoria sem modelo com proteção contra alteração de arquivos. Os testes sintéticos não validam qualidade visual. A próxima etapa de segmentação continua pendente; não existe pacote de referências seguro para iniciar a avaliação do classificador.

## Protocolo do experimento delimitado de máscaras

Usar exclusivamente desenvolvimento: casos fixos H01, H08 (halos), H09 (postura), A05 (sobreposição) e R01 (árbitro), com imagens e caixas existentes preservadas por hash. Antes de consultar a variante, revisar máscaras manuais do tronco, registrar o revisor e congelar esse gabarito de desenvolvimento. Esses casos não poderão compor a avaliação reservada.

Baseline: binarização SAM a 0,5 e descritor `masked-hsv-v3-core`. Única variante proposta: binarizar a mesma saída probabilística a 0,7, sem mudar prompt, recorte, pesos, seleção de candidatos ou limiares de distância/margem. A implementação e o resultado negativo da variante estão registrados abaixo; os executores padrão permanecem inalterados. Reutilizar a inferência probabilística para ambas as binarizações quando possível, sem dois modelos em paralelo.

Medir precisão dos pixels extraídos do tronco (`pixels corretos / pixels extraídos`) e cobertura do tronco manual (`pixels corretos / pixels do tronco manual`), por caso, incluindo perdas/abstenções. Em H01/H08/A05/R01, exigir extração não vazia, ganho médio de precisão de pelo menos 5 pontos percentuais, nenhuma piora individual de precisão e perda máxima de cobertura de 5 pontos por caso. H09 deve continuar abstendo enquanto a postura impedir tronco verificável; não vale melhorar a média aceitando H09. Mostrar também painéis original/máscara/tronco e tempo/memória, sem usar score SAM como aprovação.

Executar H01 primeiro na CPU, um processo, limite operacional de 60 s por caixa e RSS de 7 GB. Se não concluir dentro dos limites ou persistir halo visível no tronco, encerrar a variante sem ampliar. Caso passe, no máximo mais quatro caixas, uma por vez. Se os critérios agregados falharem, registrar resultado negativo e encerrar, sem varrer limiares até obter aprovação. Um resultado favorável permite nova revisão humana de candidatos, não congelamento automático nem integração à posse.

### Passo 1 executado — variante 0,7 encerrada em H01

O experimento delimitado foi implementado em `scripts/experiment_team_masks.py`, isolado dos executores padrão. A inferência real de H01 usou CPU, prompt `person` + caixa, recorte contextual 3×, checkpoint e seleção por alinhamento/score inalterados. A seleção foi feita uma única vez com máscaras a 0,5; ambas as binarizações reutilizaram exatamente a mesma matriz probabilística. No commit oficial fixado, o campo chamado `masks_logits` já contém probabilidades após sigmoid — não se aplica outra sigmoid.

Antes de consultar 0,7, o traçado manual dos troncos foi registrado em `outputs/team-mask-development-manual/manual-review.json` e congelado em `outputs/team-mask-development-frozen/frozen-development.json`, com hashes da imagem, caixas e máscaras. Revisor: Codex, inspeção visual do RGB original; limites de pixels aproximados pela baixa resolução, **sem certificação humana**. H09 foi marcado não verificável por postura. Os cinco casos foram explicitamente reservados para desenvolvimento, não para avaliação independente. O congelamento deste gabarito de desenvolvimento não aprova referências do classificador.

Resultado real em `outputs/team-mask-threshold-experiment/`:

| H01 | Baseline 0,5 | Variante 0,7 |
| --- | ---: | ---: |
| Pixels extraídos do tronco | 483 | 193 |
| Pixels corretos segundo traçado manual | 181 | 28 |
| Precisão de pixels | 37,47% | 14,51% |
| Cobertura dos 193 pixels manuais | 93,78% | 14,51% |
| Pixels extraídos fora do tronco manual | 302 | 165 |

A precisão caiu **22,97 pontos percentuais** e a cobertura caiu **79,27 pontos**. O painel `H01/comparison.png` confirma que persiste halo e que a erosão do núcleo a 0,7 remove grande parte da camisa. A revisão está em `H01/visual-review.json`. As fronteiras manuais aproximadas limitam a exatidão dos percentuais; a falha visual também impede ampliar independentemente desses números.

O processo único concluiu em **26,33 s**, com pico RSS amostrado de **6.904.463.360 bytes** (6,90 GB decimais), abaixo dos limites de 60 s e 7.000.000.000 bytes. O supervisor verifica RSS a cada 50 ms e interrompe o processo ao detectar excesso de tempo/memória; o pico é amostrado, não garantia contra excursões entre amostras. O tempo inclui inicialização, verificação dos pesos, carga, inferência e escrita. Não houve modelo paralelo ou execução MPS.

**Decisão: resultado negativo; encerrar a variante. H08, H09, A05 e R01 não foram executados.** Nenhum limiar adicional foi explorado, referência aprovada, pacote de referências congelado ou integração à posse realizada. O relatório guarda ambiente, hashes do código realmente usado e resultados; alterações posteriores no executor não reescrevem essa proveniência.

Reprodução, sempre usando saídas novas:

```sh
.venv/bin/python -m scripts.experiment_team_masks freeze \
  --manifest outputs/team-pilot-reviewed-local/initial-reviewed.json \
  --manual outputs/team-mask-development-manual/manual-review.json \
  --context-candidates outputs/team-pilot-cpu-reference-candidates/candidate-references.json \
  --out outputs/team-mask-development-frozen-new

.venv-sam3-mac/bin/python -m scripts.experiment_team_masks run \
  --truth outputs/team-mask-development-frozen-new/frozen-development.json \
  --checkpoint data/sam3-checkpoint/sam3.pt \
  --checkpoint-sha256 9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e \
  --out outputs/team-mask-threshold-experiment-new
```

O contexto copia e verifica por hash todas as máscaras anteriores, inclusive reprovadas. Elas permanecem fixas nas duas binarizações e excluem pixels compartilhados com outras pessoas. H01 é isolado e pode ser executado sem contexto; uma ampliação exige esse contexto congelado. Isso é uma exclusão conservadora condicionada a máscaras anteriores, não nova validação da segmentação de pessoas sobrepostas.

O executor só admite ampliar após H01 passar o critério numérico e uma revisão visual vinculada ao hash de `H01/result.json` declarar `halo_visible=false` e identificar `reviewer`. Uma execução aguardando revisão pode continuar com `run --resume --h01-review ARQUIVO`, mantendo `--truth`, checkpoint e `--out` originais. H01 não é repetido nessa continuação; resultados negativos não podem ser retomados. Nas quatro caixas adicionais, processadas sequencialmente, H09 é forçado a abster pela revisão manual, e o agregado exige ganho médio >=5 pontos, nenhuma piora individual e perda máxima de cobertura de 5 pontos. Um resultado favorável exige revisão humana posterior; nunca aprova referências automaticamente.

Validação do passo 1: suíte completa com 172 testes aprovada; sete testes direcionados aprovados após incluir verificações de parada negativa e exigência de revisão visual. Os testes não substituem inspeção de máscaras reais.

## Passo 2 — revisão e congelamento completo de referências

Implementado em `scripts/review_team_references.py` e `src/team_identification/reference_review.py`. O perfil `complete-v1` exige sete categorias aprovadas: jogadores de linha H/sol, H/sombra, A/sol, A/sombra, árbitro, goleiro H e goleiro A. Para goleiros, são obrigatórios `goalkeeper_verified=true`, confirmação de equipe/papel e evidência textual da identificação própria; um uniforme parecido não substitui essa verificação. Não se exige sol e sombra de cada goleiro neste passo.

Cada aprovação precisa de revisor, justificativa, `torso_verified=true`, `lighting_verified=true`, `team_and_role_verified=true`, `halo_visible=false`, hashes revisados da imagem/máscara e origem (`source_sha256`, `source_segment_id`). Esses campos de origem, equipe, papel e iluminação também são vinculados à revisão, evitando reaproveitar uma aprovação com metadados diferentes. As referências de linha continuam obrigatoriamente pre-kickoff. A aprovação explícita não substitui as verificações geométricas: vazamentos, máscaras duplicadas/fundidas, núcleo insuficiente, hashes e descritores continuam verificados no congelamento e na leitura.

O comando de revisão gera auditoria do descritor, painéis ampliados `original / SAM mask / descriptor RGB`, manifesto revisado e relatório de completude. Nenhuma inferência é necessária. Decisões aceitas: `approve`, `reject`, `pending`; deve haver uma decisão e justificativa para cada candidato, inclusive os reprovados. O manifesto original permanece intacto. Todos os candidatos são preservados como evidência para excluir pixels compartilhados. O pacote congelado copia os assets, mantém hashes e é relido para verificar os descritores e a completude.

### Resultado real em 15/11/2026

A revisão visual conservadora dos 22 candidatos atuais está em `outputs/team-reference-review-step2-decisions/visual-decisions.json`; foi aplicada em `outputs/team-reference-review-step2/`. Revisor: Codex, sem certificação humana. Foram confrontados RGB original, silhuetas e núcleos v3; os painéis individuais de H06, A03 e R02 confirmam, respectivamente, cabeça/fundo, cabeça/braço/fundo e calção dentro dos pixels usados como tronco. Os demais problemas e justificativas individuais estão registrados no arquivo de decisões. H09 permanece reprovado por postura.

**22 reprovadas, zero aprovadas, zero pendentes de decisão sobre os candidatos existentes. Nenhum pacote real foi congelado.** A completude permanece falsa nas sete categorias. Nenhum candidato atual tem papel de goleiro; o plano inicial não mostra os dois goleiros e a gravação posterior não fornece, por si só, duas referências próprias verificadas. Os metadados de origem revisados ainda ausentes não foram inventados. Não foram usadas identidades do tracking, ATD ou gabarito de posse para preencher essas lacunas.

Logo, o fluxo do passo 2 está implementado e sua revisão real foi executada, mas **o objetivo de congelar referências reais completas permanece impedido pela qualidade das máscaras e pela ausência de referências próprias verificadas dos dois goleiros**. O resultado negativo do passo 1 não foi promovido a referência nem levou a outra varredura de limiares. Para concluir, obter máscaras com tronco realmente verificável para linha H/A em sol/sombra e árbitro, além das imagens/caixas/máscaras e evidência de identificação dos goleiros H e A; revisar origem e trechos e repetir o congelamento completo.

### Comandos reproduzíveis

Criar rascunho para um conjunto novo de candidatos:

```sh
.venv/bin/python -m scripts.review_team_references \
  --manifest outputs/team-pilot-cpu-reference-candidates/candidate-references.json \
  --out outputs/team-reference-review-next-draft
```

Preencher `review-draft.json`: informar `reviewer`; revisar cada decisão, justificativa, flags e metadados. Manter `candidate_sha256` e os hashes dos assets exatamente como gerados. Novos goleiros precisam primeiro entrar como candidatos reais com suas próprias imagens, máscaras, caixas, equipe e origem verificáveis — não renomear jogadores de linha ou inventar verificação. Qualquer alteração de imagem/máscara exige nova geração e revisão vinculada aos novos hashes.

Aplicar decisões e congelar somente se o conjunto estiver completo e válido:

```sh
.venv/bin/python -m scripts.review_team_references \
  --manifest outputs/team-pilot-cpu-reference-candidates/candidate-references.json \
  --decisions outputs/team-reference-review-step2-decisions/visual-decisions.json \
  --freeze-complete --out outputs/team-reference-review-next
```

Com as decisões atuais, o resultado esperado é `incomplete_no_freeze`, com lista das sete categorias ausentes e sem pasta `frozen/`. A flag solicita o congelamento condicionado à completude; não concede aprovação às máscaras. Para um manifesto já revisado, existe também `scripts.prepare_team_references --freeze --require-complete --manifest MANIFESTO --out NOVA_PASTA`.

Compatibilidade: pacotes do fluxo antigo são identificados como `legacy-partial` e continuam utilizáveis nos experimentos anteriores, sem serem certificados como conclusão do passo 2. Pacotes `complete-v1` têm a exigência dos dois goleiros e da revisão completa verificada novamente ao carregar. O novo fluxo sempre solicita o perfil completo; omitir goleiro não é uma conclusão válida deste passo.

Validação do passo 2: **182 testes da suíte completa aprovados**, incluindo oito testes novos de completude, exigência dos dois goleiros, vínculo da revisão aos arquivos/metadados, preservação de rejeitados, portabilidade do pacote, recusa de geometria inválida e releitura do perfil completo. O congelamento exercitado nesses testes usa pixels sintéticos e não aprova referências reais. `git diff --check` sem erros.

## Ajuste autorizado: goleiros por contexto do campo e período

Em 15/11/2026, o usuário informou que o Barcelona defende à esquerda no primeiro tempo com goleiro verde, e o Athletic à direita com goleiro azul-esverdeado; no segundo tempo, os lados se invertem. A política está em `configs/team-pilot-goalkeeper-context.json`, identificada como informação fornecida pelo usuário e hipótese experimental, não como nova referência visual aprovada.

| Período do lance exibido | Gol defendido à esquerda | Gol defendido à direita |
| --- | --- | --- |
| Primeiro tempo | H / Barcelona | A / Athletic |
| Segundo tempo | A / Athletic | H / Barcelona |

`src/team_identification/goalkeeper_context.py` implementa essa regra. O verde/azul-esverdeado ajuda a estabelecer um uniforme distinto; a equipe é atribuída pelo lado defendido e período, nunca pela tonalidade isolada. Também não basta estar na metade esquerda/direita da imagem: é necessário identificar um candidato a goleiro, verificar uniforme distinto e proximidade do gol defendido, período do lance, extremo do campo e orientação da imagem. A orientação deve estar referida ao plano geral em que os lados foram confirmados, compensando eventual câmera invertida. Close, origem de replay/período desconhecidos, câmera sem orientação ou goleiro avançado sem vínculo verificável com seu gol produzem `uncertain`. Não se deriva o período a partir dos segundos da gravação nem simplesmente de um relógio maior que 45 minutos.

As cores e os lados informados ficam registrados na política; os testes comprovam a lógica e as abstenções, não a qualidade de um detector automático de goleiros. Os indícios por pessoa/lance ainda precisam ser produzidos e revisados. O módulo não detecta automaticamente verde, papéis ou gols a partir do RGB. A evidência explícita inclui `goalkeeper_candidate`, `distinctive_kit`, `near_defended_goal`, `period_verified`, `pitch_end_verified`, `orientation_verified`, `play_origin_verified`, `half`, `pitch_end`, `view=wide_pitch`, `kit_family=green/teal` e `notes`. As flags devem ser booleanas verdadeiras; desconhecimento não é convertido em aprovação.

### Dois contratos de referências

- `complete-v1`: mantém a exigência anterior de templates visuais próprios e verificados dos dois goleiros, além das cinco categorias de linha/arbitragem.
- `contextual-goalkeepers-v1`: adota a alternativa solicitada pelo usuário. Exige política de lados/período confirmada e as cinco categorias visuais H/sol, H/sombra, A/sol, A/sombra e árbitro; templates próprios de goleiro deixam de ser obrigatórios. Se uma referência visual de goleiro for incluída/aprovada, suas verificações continuam obrigatórias. Este perfil não certifica reconhecimento visual de goleiros.

A seleção do segundo perfil é explícita, usando `--goalkeeper-policy` na revisão. O pacote copia a política e sua leitura verifica novamente o contrato. O classificador visual original continua recebendo apenas RGB e máscaras. A regra contextual é aplicada separadamente e não consulta classes, papéis ou equipes do gabarito de avaliação.

Revisão real com a alternativa autorizada:

```sh
.venv/bin/python -m scripts.review_team_references \
  --manifest outputs/team-pilot-cpu-reference-candidates/candidate-references.json \
  --decisions outputs/team-reference-review-step2-decisions/visual-decisions.json \
  --goalkeeper-policy configs/team-pilot-goalkeeper-context.json \
  --freeze-complete --out outputs/team-reference-review-step2-contextual-next
```

Resultado executado em `outputs/team-reference-review-step2-contextual/`: **zero referências aprovadas, 22 reprovadas, cinco categorias visuais faltantes, nenhum pacote congelado**. A ausência de templates próprios dos goleiros deixou de bloquear essa estratégia; a qualidade das máscaras de linha/arbitragem continua impedindo concluir o congelamento. Os relatórios anteriores, que exigiam sete categorias, foram preservados como histórico.

### Aplicação contextual independente

`scripts/apply_goalkeeper_context.py` recebe detecções existentes e um JSON de contexto revisado, vinculado ao SHA256 do arquivo de detecções. Cada candidato identifica `frame`, `detection_index`, `image_sha256` e os indícios acima. O topo exige `reviewer`, `detections_sha256` e `candidates`. Os hashes impedem aplicar indícios a outra imagem ou saída sem nova revisão; não comprovam a veracidade dos indícios.

```sh
.venv/bin/python -m scripts.apply_goalkeeper_context \
  --detections CAMINHO/detections.jsonl \
  --context CAMINHO/goalkeeper-context-reviewed.json \
  --policy configs/team-pilot-goalkeeper-context.json \
  --out outputs/goalkeeper-context-next
```

A saída `contextual-goalkeepers.json` registra hipóteses por detecção, política, evidência, hashes e a decisão visual original. Divergências com a classificação visual ficam explícitas em `conflicts_with_visual`; não são resolvidas silenciosamente. Essa saída tem esquema próprio e não substitui `detections.jsonl`, não entra automaticamente nas métricas visuais v2 e não integra a posse. Não houve aplicação a detecções reais ou nova inferência nesta adaptação. Uma futura avaliação dessa estratégia deverá medir separadamente erros de identificação do goleiro, lado/período e equipe, incluindo goleiros avançados, câmeras invertidas e replays.

Validação da adaptação contextual: **189 testes completos aprovados**, incluindo os sete novos testes das quatro combinações de lado/período, abstenções, vínculo aos arquivos, preservação de decisões visuais, conflitos explícitos e congelamento/leitura sintéticos do perfil contextual sem templates de goleiros. Não mede precisão real da heurística. `git diff --check` sem erros.

## Continuação: ponte SAM/goleiros → posse e kickoff v3

A continuação está descrita em `docs/POSSE-COLETIVA.md`, seção “Continuação: kickoff sem equipe na tag e ponte visual (v3)”. Já existe entrada causal de observações SAM em `CollectiveModel.feed(..., visual_observations=...)`, sem modificar identidades/equipes ATD. O teste real do goleiro aos 00:18 executou SAM 3 em CPU e levou a posse de X para H numa cena externa, com contexto e bola nas mãos revisados manualmente. Não é detecção automática completa de goleiro/bola, nem resultado aplicado à timeline tactical.

Nenhuma das 22 referências iniciais foi promovida, o experimento negativo 0,7 permanece encerrado, e o pacote de referências de linha continua ausente. O novo teste usa uma aparição maior do goleiro e binarização SAM baseline 0,5, em uma via experimental explícita. O primeiro kickoff tactical foi corrigido pela formação no centro num instante oficial de início, porque as tags trazem equipe N/A e a bola está ausente inicialmente; o gabarito não decide a equipe. O vídeo tactical local continua necessário para validar a integração visual no mesmo recorte da simulação.

### Estado posterior ao download do tactical

O vídeo original e a homografia **agora estão presentes**, com contagem/fps/tamanho conferidos e hashes privados em `data/tactical/`. Foi concluída uma integração dirigida nos frames tactical 11689–11691: SAM 3 classificou as aparições como goleiro H pela regra contextual e o motor mudou X → H após **confirmação manual da bola nas mãos**, vinculada aos assets. O teste da bola automática com prompt `soccer ball` foi negativo, e a projeção ATD da bola não coincidiu com o objeto visível; ambos foram recusados como fonte automática de controle. Não houve alteração de limiares para obter sucesso.

Isso substitui o antigo impedimento de falta do vídeo, sem transformar o teste externo em prova tactical nem declarar cobertura visual completa. A revisão da aparição experimental não congela referências: halos superiores permanecem anotados, os núcleos foram inspecionados e os 22 candidatos antigos continuam reprovados. A execução, os comandos, os limites e as limitações atuais estão em `docs/POSSE-COLETIVA.md`, seção “Continuação com o vídeo original do Drive — integração tactical em três frames”. Os artefatos finais são `outputs/collective-sam3-tactical-verified/`, `outputs/tactical-sam3-integration-review/` e `outputs/simulacoes/tracking-gabarito-sam3-tactical-3frames/`.

### Continuação v4 — recuperação dirigida e auditoria de posse

A continuação está em `docs/POSSE-COLETIVA.md`, seção “Evidências de jogo, continuidade e recuperação visual — v4”. `scripts/run_sam3_recovery.py` executou SAM baseline em imagem inteira nos frames tactical 11518 e 12860, em CPU sequencial, respeitando 60 s / 7 GB. Foram revisadas uma pessoa de uniforme branco sem ponto ATD correspondente e uma observação ATD de equipe A projetada sobre uniforme do Barcelona. A integração é local a esses frames, com propostas/máscaras/imagens/homografia vinculadas por hash e revisão offline explícita.

A recuperação pode sustentar candidatos de recepção e a representação, mas não altera o tracking bruto, cria controle individual ou certifica classificação automática. Os três frames anteriores de goleiro continuam dependendo da revisão manual da bola nas mãos. Não se aprovaram referências de linha, não se reabriu a variante 0,7 e não se extrapolaram observações para completar a figura. O novo D usa evidência de interrupção/reinício revisada; não é inferido de bola ausente ou parada.

O recorte público mostra melhora por frame e piora no F1 oficial de segmentos, ambas registradas no relatório. Os trechos posteriores são verificações de comportamento com política congelada, sem gabarito independente de posse e sem precisão real completa do SAM. Os resultados e as limitações não autorizam afirmar reconhecimento automático integral.

### Goleiro e contexto temporal v5

A correção solicitada pelo usuário está detalhada em `docs/POSSE-COLETIVA.md`, seção v5. A46 não foi recolorido globalmente: os pontos H24/A46 alternavam entre posição do goleiro verde e localização inconsistente. Uma camada limitada à janela revisada associa indício RGB/contexto do gol H a pontos ATD existentes ou se abstém. A animação identifica G-H azul e mostra conflitos como cruz cinza. As 935 associações são propostas heurísticas com contexto de origem revisado, não revisões humanas individuais, templates aprovados ou reconhecimento automático completo. Não houve nova execução SAM nem aprovação das referências antigas.

O contexto causal de 25 frames foi testado separadamente: X cai de 534 para 463 sem a revisão do tracking. Com a correção do goleiro, X fica em 574, enquanto a acurácia sobe para 73,27% e o F1 de segmentos para 0,5524. Esse efeito misto fica explícito no relatório e na animação de três versões; não foi escondido por preenchimento indiscriminado. São resultados exploratórios do mesmo recorte público, sem gabarito nas decisões.
