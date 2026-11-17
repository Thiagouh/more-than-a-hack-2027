# More than a Hack 2027 — Ball Possession

A football analytics project exploring how computer vision and tracking data can
reconstruct ball possession throughout a match. Built around the
[More than a Hack 2027 challenge](https://github.com/metrica-sports/more-than-a-hack-2027),
this repository contains our implementation, experiments, and findings for review.

## Entrega La MasIA

A entrega integra a Frente 4 de Vitor Rocha: tracking ATD, Smart Tagging, regras
automáticas de bola morta e máquina de estados de posse. A versão integrada inclui
filtragem física da bola e eventos automáticos de controle com suporte verificado.
Não usa treinamento no jogo fornecido, correções humanas ou modelo supervisionado
de uniformes. Método, resultados e limites estão no
[documento técnico](docs/LA-MASIA-TECNICO.md).

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m scripts.build_team_submission \
  --ball-quality --include-control-events \
  --out-dir outputs/la-masia-reproduction
.venv/bin/python evaluation/validate_submission.py "outputs/la-masia-reproduction/La MasIA.csv"
```

Sem as duas novas opções, o comando reproduz a referência inicial. O diretório
de saída deve ser novo. O pacote autorizado da partida deve estar em
`data/`; ele não é redistribuído. O arquivo final já é gerado com o nome `La MasIA.csv`.

**Os experimentos históricos com uniformes rotulados neste jogo e evidências
visuais revisadas por pessoa não integram a submissão.** As
[regras oficiais atualizadas](docs/RULES.md) proíbem rótulos humanos deste jogo
inclusive para treino e validação. O [piloto YOLO](docs/YOLO-PILOTO.md) fica
registrado como pesquisa excluída da entrega atual.

## Material para revisão dos mentores

Dois documentos em português consolidam o estado atual do código e os pontos que
gostaríamos de discutir:

- [Implementações e resultados](docs/IMPLEMENTACOES-PARA-MENTORES.md): arquitetura,
  política de quatro frames em 0,6 segundo, avaliações, otimizações e reprodução.
- [Desafios e pontos de definição](docs/DESAFIOS-E-PONTOS-DE-DEFINICAO.md): desempenho para
  90 minutos, identidade, detector leve, SAM seletivo e protocolo de validação.

Os documentos de mentores preservam o retrato de 16/11/2026. A entrega atual
e o piloto YOLO de 17/11 estão nos documentos vinculados acima. SAM seletivo
permanece uma proposta; os experimentos supervisionados não integram a submissão.

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
