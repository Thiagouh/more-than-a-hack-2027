# Experimento: candidato de posse aprendido (SkillCorner open data)

Hipótese (a partir de um relato público de participante: classificadores com 100+ features treinados
em jogos abertos): um modelo treinado em outros jogos escolhe o time em posse melhor do que a regra
geométrica (mais próximo a até 3 m, margem 0,5 m, bola < 10 m/s).

**Dados:** SkillCorner open data (MIT), 11 partidas A-League baixadas (tracking de transmissão a 10 fps,
com a posse do próprio tracking como rótulo). Fora do Git (`outputs/skillcorner/`).
**Modelo:** HistGradientBoosting, 51 features por frame (sinal de proximidade, distância, velocidade da
bola e multidão em 7 offsets de -3 s a +3 s; último/próximo valor observado e suas idades; médias em 1 s e
3 s). Offsets em segundos, então vale a 10 e a 25 fps.
**Validação:** leave-one-match-out. O candidato (ML ou geométrico) passa pela **mesma** máquina de
estados da frente 4 (parâmetros reescalados ao fps) e pelo scorer oficial. Rótulo de teste: posse do
tracking com "bola solta mantém a posse".

| Média em 11 partidas | geométrico | ML |
|---|---|---|
| spellF1 | **0,3787** | 0,3220 |
| macroF1 | **0,4855** | 0,4511 |
| partidas em que o ML vence (spellF1) | | 0 de 11 |

Acurácia por frame do ML: 0,77 a 0,85. A regra geométrica, nos frames em que tem candidato, 0,80 a 0,88.

**Conclusão:** nesta formulação o modelo **não** melhora e fragmenta mais as posses. Não foi integrado à
frente 4. Limites: rótulo e tracking de outra fonte (transmissão, 10 fps, rótulo do fornecedor); sem D;
sem ajuste fino de limiar nem de hiperparâmetros. Não invalida modelos mais ricos (por exemplo, com
eventos), apenas mostra que 'mais features do mesmo sinal' não resolve.

Reproduzir: `python -m src.posse_ml.train --data outputs/skillcorner/m` (ver `src/posse_ml/`).
