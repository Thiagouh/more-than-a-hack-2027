# Referência coletiva ATD — experimento comparador

**Este comparador não é o CSV final.** A entrega usa a Frente 4 descrita em
[LA-MASIA-TECNICO.md](LA-MASIA-TECNICO.md).

## Abordagem e candidato de referência

O candidato automático de referência estima posse por frame com o tracking ATD
tactical e as marcações automáticas de início dos tempos. As observações são
convertidas para metros. O sistema verifica qualidade, proximidade, movimento
relativo e persistência antes de aceitar evidências de controle individual.

A camada coletiva usa trajetória observada da bola, contexto de 25 frames,
velocidade dos jogadores e oportunidades de recepção. Uma previsão de chegada
adversária não confirma uma troca de posse. O horizonte físico é de 2 segundos;
os parâmetros são hipóteses explícitas, sem calibração probabilística.

A aquisição e a perda usam persistência de três frames; a tolerância curta de bola
ausente é de cinco frames. Após expirar a evidência, o candidato pode emitir X.
Não há anotação manual de rótulos de posse nem leitura do gabarito durante a
geração do CSV. As posições ausentes não são preenchidas com pessoas ou bolas
inventadas. Kickoffs reiniciam o contexto; disponibilidade de evidência e instante
da observação são tratados separadamente.

## Componentes visuais excluídos do candidato

Foram usados ATD e Smart Tagging fornecidos pela competição. Vídeo e homografia
tactical foram utilizados nos experimentos visuais, sem trocar pela timeline
broadcast. A convenção permanece `video_frame = ATD_frame_id - 1`, a 25 fps.

O repositório preserva experimentos de pesquisa com SAM e reconhecimento de uniformes por histogramas HSV
e identificação temporal. O classificador congelado usa 167 exemplares de cor,
distância de Hellinger, distância máxima 0,35 e margem mínima 0,10. Combina quatro
amostras nos offsets -15, -10, -5 e 0, exigindo continuidade nos 16 frames.

Foi integrado YOLO26 nano de segmentação como alternativa leve ao SAM. A primeira
substituição completa das equipes ATD por identidades visuais foi rejeitada: no
trecho público de 120 segundos resultou em 2.859 abstenções de 3.000 frames e
spellF1 0,0444. Preservar o tracking e aplicar somente correções visuais aceitas
reproduziu os valores agregados do candidato automático nesse trecho, sem ganho
demonstrado. Outros experimentos de continuidade são documentados separadamente.

A versão atual das regras oficiais proíbe rótulos humanos do jogo fornecido,
inclusive para treino ou validação. A comunicação dos organizadores de 03/11/2026
também proíbe treino e ajuste no vídeo fornecido. Por isso, o classificador de
uniformes construído com recortes deste jogo e as evidências de goleiro/paralisação
revisadas por pessoa estão excluídos do candidato. Não há esses modelos ou
correções na cadeia que produz o CSV de referência.
SAM seletivo permanece uma proposta e não é anunciado como componente entregue.

## Avaliação do candidato automático de referência

As previsões foram geradas antes da avaliação. O único gabarito lido pelo
avaliador é o público oficial. Não foi acessado gabarito privado.

| Trecho público | Acurácia | macroF1 | spellF1 | Posses de referência / previstas |
|---|---:|---:|---:|---:|
| 120 s, frames 11185–14184 | 72,37% | 0,4750 | 0,5091 | 8 / 14 |
| 300 s, frames 11185–18684 | 59,41% | 0,5125 | 0,3467 | 17 / 28 |

O baseline publicado smooth50 apresenta spellF1 0,3256 e macroF1 0,6207 nos mesmos
300 segundos. O candidato tem maior spellF1 nessa amostra e menor macroF1. A
diferença não demonstra superioridade estatística ou generalização à partida.
A janela pública foi consultada durante o desenvolvimento e tem apenas 17 posses.

## Limitações relevantes para a entrega

O candidato de referência não possui um detector automático validado de
paralisações e não emite D nesta execução. Isso prejudica a avaliação de bola
morta. A ausência de bola é tratada como incerteza, não como prova de paralisação.
Uma hipótese simples de usar D do baseline nas abstenções melhorou a acurácia
pública, mas reduziu spellF1 de 0,3467 para 0,3333; não foi adotada.

As equipes do candidato de referência vêm dos metadados automáticos ATD, sujeitos
a erros de identidade. A geometria visual, quando experimentada, usa pés estimados
e homografia; não representa posição humana revisada. Uniforme reconhecido não
comprova controle nem bola em jogo. Há pouca validação de goleiros, sombras e
pessoas externas. Os resultados do classificador de uniforme não são acurácia
de posse.

O formato do CSV completo foi validado: 199.319 linhas, frames 0–199318 e apenas
H/A/D/X. A primeira geração levou 576,77 segundos no Mac M1 de 16 GB, com outros
processos em execução; esse tempo é do candidato baseado em tracking, não do
processamento visual da partida. A validação de formato não valida a qualidade.

## Reprodução

Com o pacote de dados autorizado em `data/`, na raiz do repositório:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m scripts.build_submission --out outputs/la-masia-reproduction
.venv/bin/python evaluation/validate_submission.py outputs/la-masia-reproduction/prediction.csv
```

O diretório de saída deve ser novo. O CSV gerado é `prediction.csv`; para o envio,
usar o mesmo conteúdo com o nome `La MasIA.csv`. O manifesto registra hashes dos
inputs e do código; a geração recusa alterações de fontes durante a execução.
O CSV não depende do modelo de uniformes, do YOLO ou de SAM. Os dados licenciados
não acompanham o clone. A implementação visual opcional está em
[YOLO-PILOTO.md](YOLO-PILOTO.md), com dependências e pesos identificados.

O repositório contém código e documentação. O modelo de uniformes treinado neste
jogo permanece entre os artefatos locais de pesquisa e não integra os arquivos da entrega. Os termos dos pesos/dependências
externos permanecem próprios e não são substituídos pela licença MIT do projeto.
