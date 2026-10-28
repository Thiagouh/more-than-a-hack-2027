# More than a Hack 2027 — Ball Possession

A football analytics project exploring how computer vision and tracking data can
reconstruct ball possession throughout a match. Built around the
[More than a Hack 2027 challenge](https://github.com/metrica-sports/more-than-a-hack-2027),
this repository will host our approach, experiments, and findings for the community.

## What we aim to build

Our goal is to identify which team has possession at each frame of a football
match, including when the ball is out of play. Beyond individual predictions, we
want to recover coherent possession sequences despite missing or unreliable ball
observations.

The project explores how player and ball tracking, match events, and video can
support this task, combining spatial context with temporal reasoning. Evaluation
focuses on both frame-level classification (**macroF1**) and the reconstruction
of possession sequences (**spellF1**).

## Coming soon

We are preparing our implementation and experiments for release. We will share
what we develop here, including our methodology, code, and results, so others can
explore the approach and build on our findings.

For now, the repository contains the challenge's starter code and evaluation
tools. Our own solution and results will be added as the project progresses.

## Background and acknowledgements

This work builds on the official
[More than a Hack 2027 repository](https://github.com/metrica-sports/more-than-a-hack-2027).
For technical background, see the [challenge description](docs/CHALLENGE.md) and
[evaluation documentation](docs/EVALUATION.md).

## License and data

Code and documentation are provided under the [MIT License](LICENSE).
Match data is subject to separate competition terms and is not distributed in
this repository.
