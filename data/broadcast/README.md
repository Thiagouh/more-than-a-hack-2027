# Coloque aqui os dados da transmissão, se for utilizá-la

Esta fonte é opcional. Cada participante obtém os arquivos pelo acesso autorizado da organização;
nenhum deles é distribuído pelo Git.

| Arquivo esperado | O que fornece |
|---|---|
| `BAR-ATH_broadcast.mp4` | Vídeo da transmissão, com cortes, aproximações e replays |
| `BAR-ATH_broadcast_atd.xml` | Metadados do tracking desta fonte |
| `BAR-ATH_broadcast_atd.txt` | Tracking automático na timeline da transmissão |
| `BAR-ATH_broadcast_homography.csv` | Transformações geométricas; parte dos frames não tem matriz válida |

O tracking possui 164.488 registros segundo o kit. Esta timeline não está alinhada à câmera tática;
não use esses índices diretamente na submissão. Nesta máquina, o contêiner do vídeo declarou
164.487 frames: a diferença deve ser investigada antes de combinar as fontes, sem aparar ou preencher
o vídeo apenas para igualar contagens. A verificação de presença não resolve essa diferença.

Execute `python scripts/verificar_dados.py --source all` na raiz.
Veja [o procedimento completo](../../docs/DADOS-LOCAIS.md).
