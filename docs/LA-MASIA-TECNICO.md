# La MasIA - tracking, física da bola e controle temporal

## Método implementado

O sistema produz H, A, D ou X por frame da câmera tactical, usando o ATD automático,
seus metadados e o Smart Tagging automático fornecidos. O comando de entrega não
requer vídeo, homografia, GPU, modelos treinados neste jogo ou correções humanas.

A base é a Frente 4 de **Vitor Rocha**, integrada do
[PR #1](https://github.com/Thiagouh/more-than-a-hack-2027/pull/1), commit
`288d5a0a1aa37324850436188655ce777263b056`. Foram preservados seus limiares de
controle e persistência. Acrescentamos filtragem física das observações da bola e
integração dos eventos automáticos da Frente 5 com suporte explícito. As opções
foram fixadas antes da nova checagem pública, sem ajustar parâmetros por esse score.

O campo mede 105,3 x 68 m. Coordenadas normalizadas são convertidas para metros.
Preservamos a timeline original de 25 fps e `video_frame = ATD_frame_id - 1`.
Slots ATD fornecem equipe automática, sem comprovar identidade permanente.

## Qualidade das observações da bola

O filtro retira coordenadas não finitas, pontos mais de 5 m além do campo e saltos
entre frames acima de 40 m/s, com dois frames adjacentes de proteção em 25 fps.
Essa duração é reescalada em testes com outro fps. Nenhuma posição é interpolada,
prevista ou inventada para preencher lacunas.

Uma regra adicional trata excursões observadas de até 2 s entre dois saltos
incompatíveis: exige contexto fisicamente estável dos dois lados e retorno a um
trajeto mais próximo e plausível entre as âncoras. É uma hipótese conservadora de
tracking temporariamente preso em outro objeto, não uma certeza de erro.

Na partida, o filtro manteve 109.293 das 113.488 observações finitas e retirou
4.195. A regra adicional de excursões não retirou pontos extras neste jogo.
Essas contagens não dizem quais posições eram corretas. Os valores mantidos são
exatamente os fornecidos; observação retirada continua ausente.

## Controle, disputas e transições

O candidato geométrico é o jogador mais próximo a até 3 m. Retiramos a candidatura
quando a margem entre times é menor que 0,5 m ou a bola supera 10 m/s. Em disputa
ou bola ausente, a posse anterior é mantida. A troca exige 12 observações do
adversário como candidato, tolerando até cinco frames sem candidato. Após D ou
início de tempo, três observações confirmam o primeiro controlador; uma reposição
pode receber pista de equipe do Smart Tagging.

A Frente 5 acrescenta aquisições automáticas com controle a até 1,5 m, margem de
1 m, velocidade relativa até 5 m/s e três frames de persistência. Sua configuração
já existente foi preservada. O evento vale apenas de `event_frame` a
`confirmed_at_frame`, sem prolongar três observações por 12 frames. Exigimos força
ordinal ao menos média, ausência de flags e suporte de bola ainda válido. Força
não é probabilidade calibrada; conflitos entre equipes ficam desconhecidos.

O bundle de eventos é ligado aos hashes dos mesmos inputs, timeline, kernel e
configuração. Foram aceitas 915 das 951 aquisições; 36 foram rejeitadas porque o
filtro retirou alguma observação do suporte. São 2.745 frames sustentados.

**O processamento é offline.** A troca confirmada é atribuída retroativamente ao
começo da sequência que a sustentou. A filtragem também pode usar frames
adjacentes futuros. Não se afirma operação causal em tempo real.

## Bola morta

D pode ser aberto pelo intervalo anterior ao kickoff de cada tempo, pelos
alinhamentos publicados; por bola mais de 0,5 m além das linhas durante três
frames; por saída além dessa margem seguida de desaparecimento; por marcação
automática de bola parada; ou por bola observada abaixo de 0,5 m/s por 100 frames.
Ausência de bola isolada não abre D. Eventos de controle não decidem D.

Após pelo menos 25 frames de D, bola observada dentro do campo a pelo menos
3 m/s confirma retomada. Há limite de 2.250 frames para um D e evidências menores
que cinco frames são descartadas. Fora dos blocos temporais publicados a saída é
X; também pode haver X enquanto falta um primeiro controle confirmado.

## Verificação da integração

O CSV completo tem 199.319 linhas, frames 0-199318, e passou no validador oficial.
O leitor e o suavizador reproduziram exatamente os quatro baselines oficiais.
A referência sem as novas opções continua reproduzível byte a byte. Os 377 testes
passaram no ambiente local e em ambiente limpo, inclusive sem fontes do Mac.

| Janela pública de 300 s | Referência inicial | Versão integrada |
|---|---:|---:|
| Acurácia por frame | 75,43% | 76,40% |
| macroF1 | 0,6819 | 0,7043 |
| spellF1 | 0,4571 | 0,4571 |
| Posses previstas / referência | 18 / 17 | 18 / 17 |

O gabarito público foi consultado depois de fixar a variante, apenas no avaliador;
não é entrada do gerador. Não foi acessado gabarito privado. A melhora ocorreu no
macroF1 e na acurácia; a métrica principal spellF1 permaneceu igual. Dezessete
posses não demonstram generalização ou precisão na partida inteira.

A saída completa contém H em 69.798 frames, A em 63.208, D em 25.459 e X em 40.854.
São previsões, não referência. O diagnóstico registra 403 períodos previstos de
posse, com duração média de aproximadamente 13,2 s.

## Teste de robustez em dados externos

Usamos três partidas da [SkillCorner Open Data](https://github.com/SkillCorner/opendata),
licença MIT, escolhidas e congeladas antes da avaliação. Nos dados originais, o
filtro não alterou os candidatos. Em 60 falhas artificiais predefinidas, nas quais
o tracking passa a seguir um adversário, os erros aceitos de candidatura caíram de
8.296 para 7.788; os acertos aceitos caíram de 45.704 para 45.612. A cobertura
passou de 57,10% para 56,46% dos frames com equipe declarada.

Esses números medem candidatos em dados externos corrompidos, não a posse final
ou D. São três jogos com rótulos automáticos do fornecedor, e os frames são
correlacionados. Falhas de quatro segundos permaneceram majoritariamente sem
solução. Não houve treinamento ou escolha de parâmetros por esses resultados.
O protocolo e os comandos de reprodução estão em
[EXPERIMENTO-QUALIDADE-BOLA.md](EXPERIMENTO-QUALIDADE-BOLA.md).

## Limites e componentes excluídos

O ATD tem lacunas, erros de equipe e posições ruidosas. Proximidade e controle
estimado não comprovam posse. Um goleiro segurando a bola pode disparar a regra de
imobilidade. Marcações de bola parada são intervalos aproximados; a retomada lenta
pode ficar atrasada. O filtro pode retirar um chute legítimo muito forte ou uma
posição válida fora do campo. Não há correção manual posterior.

O repositório preserva experimentos com SAM, YOLO e uniformes. O classificador
construído com recortes rotulados deste jogo e evidências visuais revisadas por
pessoa **não integram a submissão**. As regras atuais proíbem rótulos humanos do
jogo fornecido; a comunicação dos organizadores de 03/11/2026 também proíbe treino
e ajuste no vídeo. Não usamos esses pesos, eventos manuais ou sinal de bola
aprendido neste jogo. Os eventos integrados vêm exclusivamente do ATD automático.

## Reprodução

Ambiente verificado: Python 3.14.6, NumPy 2.5.4 e pandas 3.1.0. Na raiz do
repositório, com os dados autorizados em `data/`:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m scripts.build_team_submission \
  --ball-quality --include-control-events \
  --out-dir outputs/la-masia-reproduction
.venv/bin/python evaluation/validate_submission.py \
  "outputs/la-masia-reproduction/La MasIA.csv"
```

O diretório de saída deve ser novo. O comando regenera os eventos e cria cache
próprio. `provenance.json` registra os hashes dos inputs, configurações, código,
eventos e CSV; alterações de fontes durante a geração são recusadas.
`diagnostico.json` guarda contagens e motivos de D. Sem as duas novas opções,
o mesmo comando gera a referência inicial.

SHA256 do CSV integrado:
`4847cc74c92711fc3109094764049a3074f9530415cc1ec5b751c04866c23341`.

Dados licenciados, vídeos, caches e resultados por frame não são publicados no
repositório. O CSV é destinado ao formulário da competição. Validar seu formato
não certifica a qualidade das previsões.
