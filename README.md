# More than a Hack 2027 — Ball Possession

Projeto de análise de posse de bola em futebol, desenvolvido a partir do desafio
**More than a Hack 2027**, organizado por [Metrica Sports](https://metrica-sports.com),
[FC Barcelona](https://www.fcbarcelona.com) e
[Mobile World Capital Barcelona](https://www.mwcbarcelona.com).

O objetivo é identificar, a cada frame de uma partida profissional, se a posse
pertence ao time da casa, ao visitante ou se a bola está fora de jogo. O desafio
combina visão computacional, dados de tracking e análise temporal para reconstruir
sequências de posse mesmo quando as observações da bola estão ausentes ou inconsistentes.

## O problema

Os dados disponibilizados pela organização incluem vídeos de câmera tática e de
transmissão, tracking automático de jogadores e bola, eventos de jogo e homografias.
A saída é um CSV com uma classificação por frame:

| Rótulo | Significado |
|---|---|
| `H` | Posse do time da casa |
| `A` | Posse do time visitante |
| `D` | Bola fora de jogo |
| `X` | Estado desconhecido |

A principal métrica é **spellF1**, que avalia a recuperação de sequências completas
de posse. A **macroF1** complementa a avaliação com uma medida por frame. As definições
e os critérios estão na [documentação de avaliação](docs/EVALUATION.md).

## Estado do projeto

O repositório reúne a base oficial do desafio, os exemplos de referência e a
estrutura para desenvolvimento e avaliação de soluções. A implementação própria
ainda está em preparação; não há resultados próprios apresentados neste README.

- [`starter/`](starter/) — baselines e leitores dos dados fornecidos.
- [`evaluation/`](evaluation/) — métricas, validação de submissões e avaliação local.
- [`src/`](src/) e [`configs/`](configs/) — espaço reservado para implementação e configurações.
- [`notebooks/`](notebooks/) — exemplos de exploração dos dados.
- [`docs/`](docs/) — documentação do desafio e guias de trabalho da equipe.

## Documentação

A descrição completa do problema está em [CHALLENGE.md](docs/CHALLENGE.md).
Consulte também os [dados e seus formatos](docs/DATA.md), o
[contrato de submissão](docs/SUBMISSION.md) e as [regras](docs/RULES.md).
As orientações de ambiente, execução e colaboração ficam no
[guia da equipe](docs/GUIA-PROJETO.md).

Este repositório deriva do
[repositório oficial da competição](https://github.com/metrica-sports/more-than-a-hack-2027).

## Licença e dados

O código e a documentação seguem a [licença MIT](LICENSE).
Os dados das partidas têm licença própria, restrita aos termos da competição,
e não são distribuídos neste repositório. O acesso é disponibilizado pela organização
aos participantes inscritos que aceitam os termos de uso.
