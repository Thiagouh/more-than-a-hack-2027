# Contexto causal de 25 frames e correção do goleiro



Comparador: coletivo v4 anterior. Trecho tactical 11185–14184, 25 fps. As três configurações foram congeladas antes de consultar o gabarito; os resultados públicos continuam exploratórios.


| Medida | Antes (v4) | Só contexto 25 | Contexto + goleiro |
|---|---:|---:|---:|
| Acurácia por frame | 0.7217 | 0.7430 | 0.7327 |
| Macro F1 | 0.6527 | 0.6653 | 0.6778 |
| F1 oficial de segmentos | 0.4870 | 0.4870 | 0.5524 |
| Frames X | 534 | 463 | 574 |
| Frames D | 55 | 55 | 55 |



As lacunas X de até 25 frames entre a mesma equipe passam de 18 para 16. O contexto foi aceito em 76 frames. Reduzir X não garante acerto; comparar também os segmentos e os desacordos em `changes.csv`.



## Contexto integrado



Janela causal de 25 frames, enfatizando os cinco mais recentes: peso linear por idade e fator 2 para esses cinco. Armazena evidência direta de controle e geometria de passe; uma atribuição contextual não se transforma em nova evidência. Exige pelo menos cinco frames de evidência independente e fração ponderada de suporte ≥0,75. Esses valores são heurísticas explícitas, não probabilidades calibradas nem limiares escolhidos pelo score público.

A âncora continua sendo o último controle sustentado, com teto de quatro segundos. Bola observada válida, cobertura atual, ausência de controle/toque adversário e de interceptação iminente são requisitos. Movimento lento ou aquecimento de trajetória precisa de controle recente e companheiro observado perto da bola. D, reinício, quebra de continuidade, troca de origem e perda prolongada limpam a memória. X permanece quando esses critérios falham.



## Goleiro: causa e correção



O erro não era apenas uma cor: o ponto A46 alternava entre posição sem pessoa correspondente e o próprio goleiro verde do Barcelona, enquanto H24 às vezes ficava deslocado. Não se recoloriu A46 globalmente como Barcelona.

Em 935 frames, um indício RGB conservador do uniforme verde, restrito à região do gol defendido e ao primeiro tempo revisado, foi associado a um ponto ATD já observado. Origens escolhidas: {'24': 543, '46': 392}. A equipe H vem do contexto de gol/período confirmado, não apenas da cor. A identificação local `reviewed-gk-H` evita mudanças indevidas de equipe na identidade persistente. Nenhuma coordenada é criada ou interpolada.

Pontos conflitantes são marcados como rejeitados, ficam fora dos cálculos e aparecem como cruz cinza na animação. O goleiro H recebe G-H azul; o ponto do Athletic retomado no gol direito mantém equipe A e G-A. Quando a associação visual falha, não é inventada uma posição: permanece a abstenção da associação e o papel H previamente revisado não certifica a qualidade de toda coordenada H24.

O piloto inicial de cor aceitou uma região curta do tronco como se fosse suficiente para associar os pés. Esse caso foi reprovado no desenvolvimento; a variante final exige componente de pelo menos 20 pixels de altura e preserva as abstenções. Há 935 associações propostas no pacote final; isso não equivale a 935 revisões humanas nem a reconhecimento automático validado.

A correção é limitada à janela revisada. Não aprova referências de linha, não troca as equipes de todos os goleiros por sua posição no campo, não assume orientação de qualquer vídeo e não reutiliza as referências SAM reprovadas.



## Eventos e proveniência



Detector bruto preservado: True. O modelo corrigido usa observações com revisão de qualidade; seus eventos podem diferir dos brutos (68 eventos anteriores, 65 corrigidos). Os dois fluxos estão salvos separadamente. Portanto, a garantia anterior de eventos idênticos não é aplicada artificialmente a entradas agora corrigidas.

As imagens, a associação por frame, os limites da revisão, os hashes do vídeo/ATD/homografia e o código efetivo constam em `outputs/context25-goalkeeper-association-reviewed/` e no `protocol.json` da execução. As posições ATD originais permanecem intactas. A disponibilidade das revisões é declarada para replay offline, não demonstra latência operacional real.




**Resultado importante:** só o contexto reduz X de 534 para 463. A combinação com o goleiro rejeita evidências antes usadas e fica com 574 X, apesar de melhorar a acurácia e o F1 de segmentos. Não houve ajuste posterior dos parâmetros para esconder essa diferença.


## Conferência final

251 testes aprovados. Verificados kickoff A, três controles visuais anteriores H, 55 frames D, limite de 25 registros e ausência de continuidade contextual contra controle adversário observado. Os 1.484 frames posteriores são regressão, não validação independente.



[Simulação final de três versões](comparison.mp4) · [Proveniência](simulation-provenance.json)
