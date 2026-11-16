# Código futuro do projeto

Pasta reservada para a implementação que a equipe decidir desenvolver. Ainda não há módulos,
arquitetura ou responsáveis definidos. Essas decisões ficam para a próxima reunião.

Os scripts já fornecidos pela competição continuam em [starter/](../starter/README.md),
e a avaliação oficial está em [evaluation/](../docs/GUIA-PROJETO.md#código-e-arquivos-de-referência).

## Frente 5

`ball_events/` implementa evidências de bola, contrato de eventos v2, fila temporal de teste e ferramentas locais de validação. Consulte [o guia](../docs/FRENTE5-EVENTOS.md). Não implementa posse final.

## Frente 4

`frente4/` lê o ATD uma vez (cache NumPy), calcula features, aplica as regras de D e a máquina de
estados de posse e escreve o CSV da submissão: `python -m src.frente4 run`. Também refaz o baseline
da organização byte a byte. Consulte [o guia](../docs/FRENTE4-POSSE.md).
