# La MasIA — posse por tracking e máquina de estados

## Método entregue

O sistema produz H, A, D ou X para cada frame da câmera tactical. Usa somente o
tracking automático ATD, seus metadados e o Smart Tagging automático fornecidos
pela competição. Não requer vídeo, homografia, GPU, modelos treinados neste jogo
ou correções humanas.

A implementação principal é a Frente 4, de **Vitor Rocha**, integrada do
[PR #1](https://github.com/Thiagouh/more-than-a-hack-2027/pull/1), commit
`288d5a0a1aa37324850436188655ce777263b056`. A configuração inicial foi preservada:
nesta integração não foram ajustados parâmetros pela janela pública.

O ATD é convertido de coordenadas normalizadas para metros, com campo de
105,3 × 68 m. A timeline de 25 fps é preservada:
`video_frame = ATD_frame_id - 1`. Observações ausentes continuam ausentes.
Os slots ATD indicam a equipe automática, sem pressupor identidade permanente
de jogador.

## Controle, disputas e transições

Um candidato a controle é o jogador mais próximo, desde que esteja a até 3 m da
bola. A candidatura é retirada quando a diferença entre as distâncias dos times
é menor que 0,5 m ou a velocidade da bola supera 10 m/s. Esses filtros evitam que
um toque isolado, disputa ou passe rápido se torne imediatamente uma troca.

A máquina mantém a posse anterior em bola solta ou ausente. A troca exige 12
observações do adversário como candidato, tolerando até cinco frames sem
candidato entre elas. Após D ou início de tempo, três observações confirmam o
primeiro controlador; o Smart Tagging pode fornecer uma pista automática de time
na reposição.

**O processamento é offline.** Uma troca confirmada é atribuída retroativamente
ao começo da sequência que a sustentou. Depois de uma paralisação, a primeira
posse confirmada preenche o intervalo desde o fim de D. Não se afirma operação
causal em tempo real.

## Bola morta

O estado D pode ser aberto por:

- Intervalo anterior ao kickoff de cada tempo, usando os alinhamentos publicados
  pela organização.
- Bola observada além das linhas por mais de 0,5 m durante três frames.
- Início de uma marcação automática de bola parada do Smart Tagging.
- Bola observada a menos de 0,5 m/s por 100 frames consecutivos.

Após pelo menos 25 frames de D, uma observação da bola dentro do campo com
velocidade de pelo menos 3 m/s confirma a retomada. Há um limite de 2.250 frames
para um D. Evidências menores que cinco frames são descartadas pela máquina.
A ausência de bola, isoladamente, não abre D. Fora dos blocos temporais
publicados, a saída é X; também pode haver X enquanto falta um primeiro controle
confirmado.

## Verificação

O CSV completo tem 199.319 linhas, frames 0–199318, e passou no validador oficial.
O leitor e o suavizador reproduziram os quatro baselines oficiais exatamente,
verificando alinhamento, times, unidades e escrita dos arquivos.

| Checagem na janela pública de 300 s | Resultado |
|---|---:|
| Acurácia por frame | 75,43% |
| macroF1 | 0,6819 |
| spellF1 | 0,4571 |
| Posses de referência / previstas | 17 / 18 |

O gabarito público é aberto somente pelo avaliador, depois de gerar as previsões.
Não é entrada do gerador. Não foi usado gabarito privado. Os números são uma
checagem de integração em 17 posses, sem evidência suficiente para afirmar
generalização ou acurácia na partida inteira.

A saída da partida inteira contém H em 69.973 frames, A em 62.661, D em 25.831 e
X em 40.854. São contagens de previsões, não rótulos de referência. O diagnóstico
registra 403 períodos previstos de posse, com duração média de aproximadamente
13,2 s.

## Componentes excluídos e limites

O repositório preserva experimentos com SAM, YOLO e reconhecimento supervisionado
de uniformes. O classificador construído com recortes rotulados deste jogo e as
evidências visuais revisadas por pessoa **não integram esta submissão**. As regras
atuais proíbem rótulos humanos do jogo fornecido; a comunicação dos organizadores
de 03/11/2026 também proíbe treino e ajuste no vídeo. Não são usados pesos desses
experimentos, eventos manuais, sinal de bola aprendido no jogo ou entradas opcionais
de outras frentes no comando de entrega.

O ATD tem lacunas e erros de equipe. Proximidade não prova controle. Um goleiro
segurando a bola pode disparar a regra de imobilidade. As marcações automáticas
de bola parada são intervalos aproximados e podem abrir D cedo ou tarde. O
limiar de velocidade pode atrasar uma reposição lenta. Esses são limites das
heurísticas, sem correção manual posterior.

## Reprodução

Ambiente verificado: Python 3.14.6, NumPy 2.5.4 e pandas 3.1.0. Na raiz do
repositório, com os dados autorizados em `data/`:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m scripts.build_team_submission --out-dir outputs/la-masia-reproduction
.venv/bin/python evaluation/validate_submission.py "outputs/la-masia-reproduction/La MasIA.csv"
```

O diretório de saída deve ser novo. O comando usa a configuração fixa
`configs/frente4_v1.json`, cria um cache próprio e registra hashes dos inputs,
configuração, código e CSV em `provenance.json`. Recusa alterações de fontes
durante a geração. `diagnostico.json` guarda contagens e motivos de D.

SHA256 do CSV entregue:
`98e3d5714cb984011a5136e796b594b57f0ffb790a75307f1709a5385ccce4df`.

Os dados licenciados, vídeos, caches e resultados por frame não são publicados no
repositório. O CSV é enviado pelo formulário da competição. A validação de formato
não certifica a qualidade das previsões.
