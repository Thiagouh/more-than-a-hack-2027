# Código de referência da competição

Esta pasta mantém os scripts de referência nos caminhos originais. O algoritmo do baseline foi
preservado; as referências textuais à documentação foram ajustadas à nova organização.

| Arquivo | Função |
|---|---|
| `baseline.py` | Baseline por jogador mais próximo e suavização |
| `load_atd.py` | Loader e conferência de alinhamento do tracking |
| `load_smart_tagging.py` | Leitura dos eventos automáticos |
| `load_homography.py` | Transformações entre campo e pixels |
| `utils.py` | Desenho e animação do tracking |
| `make_dummy_submission.py` | CSVs constantes para conferir o fluxo |

Os tutoriais foram movidos para [notebooks/](../notebooks/README.md). O desenvolvimento da equipe
fica em [src/](../src/README.md). Novas previsões podem ser gravadas com
`python starter/baseline.py --out outputs/submissions/baselines`.
