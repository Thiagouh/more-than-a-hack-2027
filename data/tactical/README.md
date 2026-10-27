# Coloque aqui os dados da câmera tática

Cada participante deve obter os arquivos pelo acesso autorizado da organização. Este diretório
é o destino local; os dados não são distribuídos pelo Git.

| Arquivo esperado | O que fornece |
|---|---|
| `BAR-ATH_tactical.mp4` | Vídeo de referência; cada linha da submissão corresponde a um frame dele |
| `BAR-ATH_tactical_atd.xml` | Metadados para interpretar equipes e slots do tracking |
| `BAR-ATH_tactical_atd.txt` | Posições e velocidades automáticas de jogadores e bola |
| `BAR-ATH_tactical_homography.csv` | Matrizes que relacionam posições no campo e pixels |
| `BAR-ATH_tactical_smart_tagging.xml` | Intervalos de fases e eventos automáticos |

O contrato da submissão tem 199.319 frames, a 25 fps. No ATD entregue, `video_frame = frame_id - 1`;
os tempos dos eventos são segundos desde o começo do vídeo. Slots não são identidades estáveis de
jogadores, e bola ausente no tracking não significa automaticamente bola parada.

Execute `python scripts/verificar_dados.py` na raiz para conferir os arquivos esperados.
Veja [o procedimento completo](../../docs/DADOS-LOCAIS.md).
