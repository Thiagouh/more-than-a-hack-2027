# Frente 2: trilha da bola

Missão: reduzir o problema da bola que some. Tudo é gerado por código, sem nenhuma correção
manual de posição e sem usar o gabarito.

## Entrega

```bash
pip install -r requirements-bola.txt
python -m src.bola.entrega        # outputs/bola/sinal_bola.csv (~1,5 min) + conferência da nota
```

```python
from src.bola.entrega import carregar_sinal
sinal = carregar_sinal()          # 199.319 linhas, uma por frame do vídeo tático
```

| coluna | significado |
|---|---|
| `frame` | frame do vídeo tático (0 a 199318), igual à linha da submissão |
| `x`, `y` | posição da bola em metros (campo de 105,3 × 68) |
| `fonte` | `observado`, `preenchido` (buraco ≤ 0,5 s, reta), `estimado` (buraco longo, modelo A) ou `ausente` |
| `buraco_id` | −1 fora de buraco; senão, a linha correspondente em `outputs/bola/buracos.csv` |
| `time_proximo` | H ou A: time do jogador mais próximo, só onde a bola é vista |
| `dist_h`, `dist_a` | distância em metros do jogador mais próximo de cada time até a bola, vista ou estimada |
| `time_antes`, `time_depois` | nos buracos, o time mais próximo nos 25 frames vistos antes e depois |
| `saiu_campo` | o buraco começou com a bola fora das linhas (pista de D) |
| `prob_h` | nos buracos: P(time mais próximo = H), pelo modelo A |
| `posse_buraco` | **sugestão para o buraco**: D se `saiu_campo`, senão `time_antes` |
| `posse_buraco_modelo` | alternativa: troca de time no ponto indicado pelo modelo A |

**Para a frente 4:** usar `time_proximo` onde a bola é vista e `posse_buraco` onde ela some, com
a mesma suavização de ±2 s do baseline. Isso leva o spellF1 do trecho público de **0,3256 para
0,4205** (`entrega.rotulos_exemplo` mostra como). A máquina de estados deve fazer melhor do que
essa suavização, mas manter a posse durante o buraco é o ponto principal.

**Atenção:** a posição `estimado` é uma região, não um ponto. O erro mediano é de cerca de 9 m. Não
decida a posse pelo jogador mais próximo dessa posição: no teste, isso piorou a nota (0,3442).

## O que descobrimos

1. **A bola some em 41% dos frames avaliados, quase sempre por muito tempo.** Depois da limpeza
   sobram 258 buracos. Os 135 que duram mais de 5 s concentram 95% dos frames sem bola. Os 110
   buracos de 0,5 a 2 s vêm quase todos das falhas removidas. Preencher buracos curtos quase não
   muda nada.
2. **O tracker às vezes "desliza" de 30 a 50 m em menos de 1 s.** Acima de 40 m/s, a posição vira
   ausente (1.260 frames avaliados).
3. **Nos buracos, manter a posse é o que mais ajuda.** Em 60% dos buracos o time antes e o time
   depois são o mesmo. Quando a bola some depois de sair do campo (15% dos buracos), o certo é D.
4. **Modelo A** (`modelo.py`): gradient boosting que aprende, nos frames em que a bola foi vista,
   onde ela está a partir dos jogadores. Treina em buracos sintéticos, em que apagamos trechos
   vistos. Prevê o desvio em relação à interpolação e espelha o campo no 2º tempo, porque os
   times trocam de lado.

   | erro mediano de posição | ≤10 s | 10–20 s | >20 s | todos |
   |---|---|---|---|---|
   | interpolação | 6,4 m | 12,4 m | 21,8 m | 16,0 m |
   | regra dos jogadores | 7,4 m | 11,6 m | 19,1 m | 14,3 m |
   | **modelo A** | 7,7 m | **8,6 m** | **9,5 m** | **9,0 m** |

5. **Ainda não está decidido onde fica a troca dentro de um buraco** (`troca.py`). O teste vale
   para os buracos em que o time antes é diferente do time depois:

   | estratégia | buracos sintéticos | spellF1 público |
   |---|---|---|
   | troca no fim, quando a bola reaparece | 51,9% | **0,4205** |
   | troca no meio | 61,4% | 0,3795 |
   | troca pelo modelo A | **71,5%** | 0,3692 |

   O trecho público tem **um único** buraco com troca, então não dá para concluir com ele. A
   validação em outros jogos (frente 5) pode decidir. Até lá, a sugestão é `posse_buraco`.

6. **Detector no vídeo** (`detector.py`, `notebooks/05_detector_bola_colab.ipynb`): ver a seção
   abaixo.

## Detector no vídeo

Teste em 13/11: YOLO11m pré-treinado (classe "sports ball"), em CPU, com imagem de 1920 px. A
amostra é pequena: 30 frames de controle e 32 de buraco do trecho público.

| | resultado |
|---|---|
| controle: alguma detecção a ≤ 20 px da bola do tracker | 57% |
| controle: alguma detecção a ≤ 40 px | 73% |
| buracos: alguma detecção dentro do campo | 81% |
| detecções por frame (média) | **5,4** |

**O detector vê a bola, mas também vê muitas outras coisas.** Com 5 a 6 candidatos por frame
(cabeças, chuteiras, marcas no gramado), os 81% dos buracos não significam que a bola foi
recuperada: só mostram que há candidatos. Para servir, falta escolher o candidato certo. Os
caminhos possíveis são exigir continuidade de um frame para o outro (a bola não pula), filtrar
pela região que o modelo A prevê e fazer ajuste fino com rótulos automáticos (posições do tracker
projetadas no vídeo, sem anotação manual).

Custo: cerca de 3 a 7 s por frame em CPU. O jogo inteiro só é viável em GPU, pelo
`notebooks/05_detector_bola_colab.ipynb`.

## Arquivos

| arquivo | o que faz |
|---|---|
| `carregar.py` | lê o ATD e guarda um cache NumPy; provisório até a frente 1 entregar a função oficial |
| `limpar.py` | remove falhas, preenche buracos curtos, monta `buracos.csv` |
| `estimar.py` | regra simples para os buracos longos (interpolação + aglomerado de jogadores) |
| `modelo.py` | modelo A: treino, validação cruzada por tempo e previsão nos buracos reais |
| `troca.py` | onde colocar a troca de posse dentro de um buraco |
| `entrega.py` | junta tudo em `sinal_bola.csv` |
| `avaliar_estimador.py` | erro das regras simples em buracos sintéticos |
| `experimento_buracos.py` | efeito de cada tratamento de buraco na nota (trecho público) |
| `video_analise.py` | vídeo dos 5 min com gabarito, tracking projetado e bola limpa |
| `detector.py` | experimento com YOLO pré-treinado nos buracos |

Os números das notas usam o trecho público, que tem só 17 posses. Eles indicam a ordem de grandeza
e não devem ser usados para escolher parâmetros (ver `docs/EVALUATION.md`).
