# More than a Hack 2027 — Ball Possession

A football analytics project exploring how computer vision and tracking data can
reconstruct ball possession throughout a match. Built around the
[More than a Hack 2027 challenge](https://github.com/metrica-sports/more-than-a-hack-2027),
this repository contains our implementation, experiments, and findings for review.

## Material para revisão dos mentores

Dois documentos em português consolidam o estado atual do código e os pontos que
gostaríamos de discutir:

- [Implementações e resultados](docs/IMPLEMENTACOES-PARA-MENTORES.md): arquitetura,
  política de quatro frames em 0,6 segundo, avaliações, otimizações e reprodução.
- [Desafios e pontos de definição](docs/DESAFIOS-E-PONTOS-DE-DEFINICAO.md): desempenho para
  90 minutos, identidade, detector leve, SAM seletivo e protocolo de validação.

O detector leve com SAM seletivo ainda é uma proposta. A nova acurácia de posse
com a identificação visual atual permanece pendente; os documentos distinguem
esse resultado da avaliação de uniformes e das medições anteriores de posse.

## What we aim to build

Our goal is to identify which team has possession at each frame of a football
match, including when the ball is out of play. Beyond individual predictions, we
want to recover coherent possession sequences despite missing or unreliable ball
observations.

The project explores how player and ball tracking, match events, and video can
support this task, combining spatial context with temporal reasoning. Evaluation
focuses on both frame-level classification (**macroF1**) and the reconstruction
of possession sequences (**spellF1**).

## Implementation status

The repository includes causal tracking events, collective possession inference,
supervised uniform classification, temporal identity checks, visual overlays,
evaluation tools, and automated tests. The selected supervised uniform policy
uses four samples over 0.6 seconds, checking continuity in every intervening frame.

These are experimental implementations. Results from short public windows and
manually reviewed crops do not establish full-match accuracy or real-time speed.
Match videos, trained local artifacts, and generated outputs are not distributed
in this repository. See the [documentation index](docs/README.md) and the
[mentor review documents](docs/IMPLEMENTACOES-PARA-MENTORES.md) for instructions
and the exact scope of the available evidence.

## Background and acknowledgements

This work builds on the official
[More than a Hack 2027 repository](https://github.com/metrica-sports/more-than-a-hack-2027).
For technical background, see the [challenge description](docs/CHALLENGE.md) and
[evaluation documentation](docs/EVALUATION.md).

## License and data

Code and documentation are provided under the [MIT License](LICENSE).
Match data is subject to separate competition terms and is not distributed in
this repository.
