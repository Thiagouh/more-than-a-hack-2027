# Tutoriais e exploração

Os três notebooks são os tutoriais originais, movidos de `starter/` com ajuste de caminhos:

1. [Smart Tagging](01_smart_tagging.ipynb): eventos automáticos e intervalos.
2. [Tracking ATD](02_tracking_atd.ipynb): leitura, coordenadas e visualização.
3. [Primeira submissão](03_first_submission.ipynb): baseline, CSV e avaliação.

Na raiz do repositório, com `.venv` ativo:

```bash
python -m pip install -r requirements-starter.txt
jupyter notebook
```

Os imports encontram a raiz tanto a partir dela quanto desta pasta. As previsões do notebook 03
são gravadas em `outputs/submissions/`. As saídas antigas das células foram limpas para não
apresentar resultados de uma execução anterior como atuais.

Comparações com o ground truth público nos exemplos servem para compreender e verificar o
pipeline. Não escolher versões ou ajustar parâmetros a partir dessas notas.
