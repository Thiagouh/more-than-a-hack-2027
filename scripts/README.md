# Ferramentas locais

`verificar_dados.py` confere se os arquivos esperados existem e não estão vazios, usando os
mesmos caminhos de `games.py`. Não depende de numpy, pandas ou kloppy e não baixa nem altera assets.

```bash
python scripts/verificar_dados.py
python scripts/verificar_dados.py --source all
python scripts/verificar_dados.py --data-dir /caminho/para/meus-dados
```

O retorno é 0 quando todos os arquivos solicitados estão presentes, ou 1 se algum está ausente
ou vazio. O padrão confere apenas a fonte tática; broadcast é opcional.
Veja [o guia de preparação](../docs/DADOS-LOCAIS.md).
