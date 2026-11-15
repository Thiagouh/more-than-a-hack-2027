# Piloto de equipes com SAM 3

Classificação independente por aparição: H = Barcelona, A = Athletic Club, referee = árbitro e uncertain = incerto. Sem identidade persistente, números, ATD, contagem esperada ou gabarito de posse. Distância e margem são evidências, nunca probabilidades.

## Estado observado

A implementação local foi exercitada com testes sintéticos, adaptador SAM simulado e inferência real no Mac. O checkpoint oficial foi obtido após aprovação e autenticação do usuário; SHA256 verificado: 9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e. CPU concluiu o primeiro recorte com máscara não vazia; MPS concluiu uma tentativa sem candidatos e apresentou esperas prolongadas em outras tentativas, interrompidas explicitamente. **MPS ainda não é considerado estável.** Precisão e cobertura reais ainda não foram medidas; nenhuma integração à posse foi feita.

Os materiais reais permanecem em outputs/, ignorado pelo Git. O draft original foi preservado. A cópia outputs/team-pilot-reviewed-local/initial-reviewed.json contém revisão visual das 20 caixas de jogadores e duas de árbitros. A03/A04 foram ampliadas para incluir os pés; a nova imagem está em outputs/team-pilot-reviewed-local/box-review/initial-box-review.png. Há outros oficiais distantes/parciais fora dessas referências selecionadas. H09 está curvado e deve permanecer reprovado como referência enquanto seu tronco não for verificável. Sol foi identificado em H06/H07 e A05/A06; os demais candidatos foram marcados sombra. Confirmar a iluminação efetivamente coberta pelo tronco na revisão das máscaras.

A gravação de tela tem aproximadamente 32 s e cortes/saltos de reprodução. Os arquivos 16.png até 31.png são amostras posteriores; seus nomes e timestamps representam segundos da gravação, não frames nem tempo original da partida. Não há sincronização com o tracking.

## Contratos e proteção contra referências inválidas

- O descritor usa todos os pixels RGB válidos do tronco: 20–55% da altura e 20–80% da largura da máscara original. A exclusão das sobreposições acontece depois, sem recalcular o enquadramento.
- Critérios provisórios: pelo menos 24 pixels, pelo menos 50% do tronco original restante, máscara com altura >=20 e largura >=10. Silhuetas com largura/altura >0,85 abstêm por postura/fusão suspeita.
- Duplicatas com IoU >=0,85 e contenção >=80% tornam ambas as detecções incertas; elas permanecem nos diagnósticos. Essas regras são heurísticas: não garantem detectar toda fusão conectada ou todo tronco incorreto.
- Aprovação das caixas é separada de aprovação das máscaras. A geração produz apenas candidate-references.json, com mask_reviewed=false e approved=false para cada candidato.
- O congelamento exige aprovação explícita, reviewer, árbitro, H/A de linha em sol e sombra, origem pre-kickoff dos jogadores de linha e goalkeeper_verified=true para cada referência de goleiro.
- Todos os candidatos, inclusive reprovados, participam da remoção de pixels compartilhados. Máscaras que escapam mais de 10% da caixa revisada são recusadas ao congelar.
- O pacote congelado copia imagens/máscaras para assets/, guarda hashes e descritores e pode ser transferido integralmente. A leitura verifica hashes e recalcula os descritores.
- Os limiares provisórios são distância máxima 0,35 e margem mínima 0,10. Não foram calibrados com os frames de avaliação. Caso seja necessário ajustá-los, separar um conjunto de desenvolvimento e reservar nova avaliação.
- Goleiros sem referência aprovada não têm template próprio. Aparências distantes/ambíguas abstêm; um classificador de cor isolado **não garante** distinguir um goleiro cujo uniforme se pareça com outra classe. Essa falha deve ser medida na avaliação; o piloto não está validado para goleiros.
- Entradas aceitam RGB e RGBA totalmente opaco, máscaras binárias 0/1/255 com dimensões exatas, caixas finitas dentro da imagem e rótulos seguros únicos.

## Comandos reproduzíveis

Executar na raiz do projeto. Usar uma pasta nova para cada saída; nenhuma etapa sobrescreve resultados existentes.

Revisão local, sem modelo:

```sh
python -m scripts.prepare_team_references \
  --manifest outputs/team-pilot-reviewed-local/initial-reviewed.json \
  --out outputs/team-pilot-box-review-next --boxes-only
```

No host externo, instalar PyTorch/CUDA conforme o README oficial e instalar o SAM 3 a partir de checkout Git limpo no commit fixado. Obter o checkpoint pelos meios autorizados; os executores não baixam pesos. Transferir o projeto, manifesto revisado e imagem, preservando os caminhos relativos. Não é necessário transferir o broadcast completo de 14 GB.

```sh
git clone https://github.com/facebookresearch/sam3.git /opt/sam3
git -C /opt/sam3 checkout --detach eef6839bcfc3458dd52ea80535dc6bd62aea86b1
python -m pip install -e /opt/sam3
export SAM3_CHECKPOINT=/modelos/sam3.pt
export SAM3_SHA256=SUBSTITUIR_PELO_SHA256_AUTORIZADO_DE_64_HEXADECIMAIS

python -m scripts.prepare_team_references \
  --manifest outputs/team-pilot-reviewed-local/initial-reviewed.json \
  --checkpoint "$SAM3_CHECKPOINT" --checkpoint-sha256 "$SAM3_SHA256" \
  --out outputs/team-pilot-gpu-candidates
```

A inicialização verifica o hash antes de importar o modelo e exige CUDA, commit fixado e checkout sem alterações rastreadas. O adaptador usa caixas normalizadas cxcywh e máscaras booleanas conforme o [código oficial do commit fixado](https://github.com/facebookresearch/sam3/blob/eef6839bcfc3458dd52ea80535dc6bd62aea86b1/sam3/model/sam3_image_processor.py). Para cada caixa, prefere alinhamento espacial e depois score; ausência de candidatos gera máscara vazia para revisão, nunca referência válida.

Inspecionar initial-mask-review.png e cada máscara candidata. Em candidate-references.json, preencher reviewer, marcar mask_reviewed=true e approved=true **somente** nas referências verificadas. Reprovar fusões, postura inadequada, pouco tronco e máscaras erradas. Não aprovar H09 automaticamente.

```sh
python -m scripts.prepare_team_references \
  --manifest outputs/team-pilot-gpu-candidates/candidate-references.json \
  --freeze --out outputs/team-pilot-frozen

python -m scripts.prepare_team_evaluation \
  --references outputs/team-pilot-frozen/frozen-references.json \
  --frames-dir outputs/team-pilot-initial/screen-recording-frames \
  --recording '/dados/Gravação de Tela 2026-11-14 às 21.12.20.mov' \
  --out outputs/team-pilot-evaluation-inputs

python -m scripts.run_team_pilot \
  --manifest outputs/team-pilot-evaluation-inputs/run-manifest.json \
  --checkpoint "$SAM3_CHECKPOINT" --checkpoint-sha256 "$SAM3_SHA256" \
  --out outputs/team-pilot-gpu-results
```

O preparador seleciona por padrão segundos 16,19,22,25,28,31. O manifesto registra hashes, separação evaluation e semântica dos timestamps. O executor valida todas as entradas antes de carregar o modelo. Saídas: frames/, masks/, detections.jsonl com contagens internas, motivos e áreas válidas; provenance.json com manifesto, referências, hashes, commit e ambiente GPU. Não desenha contagens nem usa resultados anteriores para decidir o time.

## Avaliação visual independente

Preencher visual-truth-draft.json sem consultar as decisões do classificador, incluindo **todas** as pessoas visíveis, goleiros e árbitros. Cada pessoa precisa de label local ao frame, box xyxy, classification (H/A/referee/uncertain), evaluable boolean, lighting (sun/shade/mixed) e occlusion (isolated/overlapped). A marca evaluable indica uniforme visualmente avaliável; árbitros participam da precisão, mas não do denominador de cobertura de atletas. Depois, preencher reviewer e reviewed=true.

```sh
python -m scripts.evaluate_team_pilot \
  --truth outputs/team-pilot-evaluation-inputs/visual-truth-draft.json \
  --detections outputs/team-pilot-gpu-results/detections.jsonl \
  --out outputs/team-pilot-evaluation-report
```

Associação de caixas um a um, independente da classe, por cardinalidade máxima com IoU >=0,5 e preferência determinística local por maior IoU. Detecções extras permanecem no denominador de precisão; extras com associação possível são duplicatas. Pessoas sem associação são perdas. Anotações uncertain não contam como acerto de decisão aceita.

Precisão = decisões aceitas corretas / todas as decisões aceitas. Cobertura = atletas avaliáveis associados a decisão aceita / todos os atletas avaliáveis, incluindo os perdidos. O relatório inclui incertos, duplicatas, perdas, quantidade de frames e estratos por iluminação/sobreposição. Precisão dos estratos usa pessoas associadas; extras não associados aparecem apenas no total.

Metas iniciais: precisão >=95% e cobertura >=80%. Seis frames são apenas avaliação exploratória: ampliar para pelo menos 30 frames independentes e sequências difíceis quando pixels adicionais estiverem disponíveis. Mesmo um relatório acima das metas não autoriza automaticamente integração à posse. Não usar o gabarito de posse para ajustar uniformes ou limiares.

## Testes locais

```sh
.venv/bin/python -m unittest tests.test_team_pilot tests.test_team_pilot_safety
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

Esse comando mede carga, inferência, memória do processo e memória MPS; salva single-box-mask.png para inspeção. Só depois de verificar funcionamento, consumo de memória e máscara, executar as caixas completas com .venv-sam3-mac/bin/python -m scripts.prepare_team_references, os argumentos já documentados e **--device mps**. O mesmo argumento está disponível em scripts.run_team_pilot. O padrão continua CUDA por compatibilidade. --device cpu é explícito e experimental.

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
