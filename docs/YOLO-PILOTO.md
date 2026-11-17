# Piloto YOLO — La MasIA

**Experimento de pesquisa excluído da submissão.** O classificador de uniformes
foi construído com recortes rotulados deste jogo. As regras atuais e a comunicação
dos organizadores de 03/11/2026 impedem seu uso na entrega. O CSV automático
de referência não depende dele. Nenhum modelo derivado desses rótulos será
publicado como parte da entrega.

## Implementação

O executor `scripts.run_team_identity` aceita `--backend yolo`, mantendo SAM como
padrão anterior. O novo adaptador exige segmentação de instâncias e classe zero
`person`. Caixas não são convertidas artificialmente em silhuetas. Os polígonos
são rasterizados nas coordenadas da imagem original antes da geometria existente.

O classificador supervisionado v4, os limiares 0,35/0,10, a política de quatro
amostras em 0,6 s e os critérios de identidade permanecem iguais. Não há entrada de
gabarito no executor. A identificação é calculada a cada frame, sem pular frames.

Modelo experimental: `yolo26n-seg.pt`, SHA256
`361fbfabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f`.
Ultralytics 8.4.215, PyTorch 2.10.0, torchvision 0.25.0, MPS no Apple M1.
Dependências diretas em `requirements-yolo.txt`. FFmpeg é necessário.

## Reprodução local

Com o vídeo tactical, o ATD, a homografia e o modelo supervisionado congelado:

```sh
python -m scripts.run_team_identity \
  --backend yolo --checkpoint yolo26n-seg.pt \
  --supervised-model outputs/uniform-supervised-model-v4-four-frames/model.json \
  --device mps --imgsz 640 --start-frame 11160 --end-frame 14185 \
  --no-save-masks --out outputs/yolo-public-reproduction
```

O diretório de saída deve ser novo. O checkpoint público é resolvido pelo pacote;
seu hash e a versão do pacote ficam no manifesto. O modelo local de uniformes
não acompanha o clone e não integra a submissão da competição. Nenhum vídeo,
recorte ou tracking derivado é adicionado ao repositório.

## Medições iniciais em 17/11/2026

| Configuração | Frames | Média do processamento/frame | Tempo do intervalo inteiro |
|---|---:|---:|---:|
| YOLO nano-seg, 960, antes da nova otimização espacial | 50 | 0,623 s | 42,98 s |
| YOLO nano-seg, 640, depois da otimização espacial | 50 | 0,153 s | 20,55 s |

A média de processamento inclui segmentação e pipeline, mas exclui leitura do
frame e serialização final. O tempo do intervalo inclui decodificação e escrita,
mas começa depois da construção do modelo e dos hashes iniciais. A diferença
entre as linhas mistura resolução e otimização: não mede isoladamente nenhuma
delas. Esses cinquenta frames não sustentam previsão de tempo da partida inteira.

A configuração 640 detectou menos pessoas e produziu menos classificações
aceitas nesse trecho. Sua adoção depende da avaliação de posse. Os resultados
de uniforme não equivalem a acurácia de posse.

## SAM seletivo e limites

SAM seletivo ainda não foi implementado. Primeiro é medida a versão YOLO isolada.
O novo detector pode perder atletas pequenos ou fundir pessoas; a base da máscara
continua sendo uma estimativa de pé. A qualidade dos recortes pode diferir dos
recortes supervisionados. Identidade ambígua e uniforme incerto continuam podendo
produzir abstenção.

O pacote e os pesos Ultralytics têm seus próprios termos AGPL-3.0/Enterprise,
conforme a documentação do fornecedor; a licença MIT do projeto não os substitui.

Fontes oficiais: [YOLO26](https://docs.ultralytics.com/models/yolo26/) e
[inferência](https://docs.ultralytics.com/modes/predict/).

## Resultado do primeiro experimento de posse

Na comparação de 3.000 frames públicos (11185–14184), substituir todas as equipes
ATD pelas identificações visuais do YOLO nano-seg 640 resultou em acurácia 4,70%,
macroF1 0,0376 e spellF1 0,0444: 2.859 abstenções e 141 respostas H. A adoção dessa
substituição foi rejeitada. O comparador histórico com evidências visuais revisadas
teve spellF1 0,5524, mas não é uma solução automática disponível para a partida.

A execução completa dos 3.025 frames com aquecimento levou 444,04 s, média de
processamento 0,1379 s/frame e p95 0,2162 s/frame. Houve outros processos locais,
inclusive pilotos breves no mesmo backend; isso limita a interpretação temporal.

Foram observadas 2.877 identidades locais, com média de 11,19 observações por
identidade. A janela de uniforme exige 16 frames consecutivos; 14.730 detecções
ficaram em aquecimento. São diagnósticos do desenvolvimento, não validação
independente. Uma alternativa explícita de continuidade por sobreposição das
caixas está em avaliação (`--identity-method image-overlap`), com IoU mínimo 0,3,
margem global 0,1 e reinício após lacunas ou associação ambígua.

A política opcional `--identity-policy correct-teams` do avaliador preserva o ATD
como fonte declaradamente não verificada e aplica apenas equipes visuais aceitas
com associação um a um. Não recupera pessoas nem altera coordenadas. Mudança de
equipe cria novo namespace de track para não herdar controle individual anterior.
Essa política é um experimento separado da substituição estrita, que continua
sendo o padrão.

## Variante de caixas e verificação da entrega

O backend opcional `yolo-boxes` exige pesos de detecção, utiliza caixas observadas
sem criar máscaras artificiais e registra `estimated_box_bottom` e
`uncalibrated_box_approximation`. A projeção continua sendo estimada. O piloto
nano 640 de 200 frames levou 89,69 s, em condição com outros processos de visão;
portanto não há benchmark isolado ou resultado de posse dessa variante.

As execuções adicionais de continuidade foram interrompidas após a conferência
das regras. O modelo de uniformes baseado neste jogo foi retirado dos artefatos
preparados para publicação. Não há execução visual de partida inteira em andamento.
