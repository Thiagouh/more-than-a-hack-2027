# Desafios e decisões em aberto — revisão dos mentores

**Retrato do projeto em 16/11/2026.** Leitura complementar a [Implementações para os mentores](IMPLEMENTACOES-PARA-MENTORES.md), que descreve o código e os resultados existentes. Este documento apresenta limites, hipóteses e perguntas; não atribui resultados a componentes ainda não implementados.

## 1. Decisão central: tornar a visão viável para uma partida

O fluxo atual executa SAM completo em cada frame, estima geometria a partir das máscaras e identifica uniformes por exemplos de cor. É útil para investigação, mas caro demais no Mac disponível para escalar diretamente aos 90 minutos.

O objetivo inicial escolhido é **processamento offline em algumas horas**, tomando até seis horas como alvo de planejamento, sem alegar que já seja atingível. A primeira referência de hardware é o Apple M1 de 16 GB disponível. Não foi definida outra GPU, contratado serviço de nuvem ou validado processamento ao vivo.

O usuário concordou com a direção **modelo leve no fluxo principal e SAM nos casos difíceis**. O trabalho nessa arquitetura foi suspenso para esta revisão com mentores. Não existe ainda novo detector treinado/instalado, política de escalonamento para SAM, cache completo de visão ou resultado comparativo do sistema híbrido.

### Ordem de grandeza

90 minutos a 25 fps correspondem a **135.000 frames**; o vídeo da gravação, com intervalos e aquecimento, pode ser maior. Para processar 135.000 frames em seis horas, a média de **toda a cadeia** deve ficar em até **0,16 s/frame**, ou **6,25 frames/s**. São requisitos aritméticos, não benchmarks.

| Medida atual | Evidência | Implicação |
|---|---|---|
| CPU espacial: 2,025 → 0,561 s/frame | Equivalência em 20 frames reais | Melhorou 3,61×, mas essa etapa isolada já excede a média alvo de 0,16 s |
| SAM: aproximadamente 9,07 s/frame | Apenas dois frames reais, medidos por etapas | Não cabe como dependência obrigatória por frame |
| Execução inteira | Novo benchmark completo não concluído | Não extrapolar 3,61× como ganho total |
| Precisão reduzida | Tentativa sem ganho consistente | Não adotada como solução |

A estimativa inicial de muitas horas para dois minutos veio do progresso de uma execução, enquanto o teste por etapas ocorreu em outra condição de carga/aquecimento. Eles não são medições comparáveis. Projeções para a partida inteira devem registrar essa incerteza, decodificação, inicialização, memória e escrita em disco.

**Pergunta aos mentores:** qual orçamento de tempo, memória e qualidade é razoável para a primeira versão offline, e quais componentes devem receber esse orçamento?

## 2. Detector leve: como escolher sem perder atletas pequenos

As imagens táticas incluem pessoas pequenas, perfis, corrida, oclusões e aglomerações. Reduzir a resolução pode acelerar a rede e, ao mesmo tempo, remover a informação necessária para detectar o atleta ou distinguir seu tronco.

É necessário escolher e comparar modelos que entreguem caixas, máscaras ou pontos anatômicos úteis. O fluxo atual estima pés a partir da máscara; substituir por caixa não é um detalhe de implementação. O centro inferior de uma caixa não é um pé observado e não deve ganhar automaticamente o mesmo nível de confiança.

| Decisão pendente | O que precisamos medir |
|---|---|
| Detector de caixas, segmentação leve ou pose | Recall de pessoas pequenas, duplicatas, fusões, qualidade de pés e custo |
| Resolução inteira, regiões ou mosaicos | Perdas perto das bordas, pessoas parcialmente cortadas e custo acumulado |
| CPU, MPS ou outro backend | Tempo real no hardware alvo e equivalência de resultados |
| Frequência da rede | Erros entre detecções, latência e custo do acompanhamento intermediário |
| Pesos públicos ou ajuste específico | Necessidade de rótulos, separação dos conjuntos e risco de sobreajuste |

O repositório é MIT; novas dependências e pesos podem ter outros termos. A escolha ainda precisa documentar licença, origem, versão e hash dos pesos antes de redistribuição. Nenhum modelo candidato deve ser chamado de aprovado apenas por ser pequeno ou rápido em um benchmark externo.

**Perguntas aos mentores:** qual família de modelo seria adequada para pessoas pequenas nesta câmera? É preferível preservar máscaras leves para reduzir mudanças no pipeline ou aceitar uma nova estimativa de geometria por caixas/pose?

## 3. Acionar SAM somente quando houver uma pergunta específica

O SAM não deve ser acionado automaticamente a cada resultado Incerto de uniforme. A incerteza pode ser causada por iluminação, escala, falta de exemplos, recorte parcial, identidade perdida ou geometria inválida; uma máscara mais cara não resolve necessariamente essas causas.

Possíveis gatilhos a discutir — **ainda não implementados**:

| Situação | Pergunta que o SAM poderia ajudar a responder | Limite |
|---|---|---|
| Duas pessoas sobrepostas | Há silhuetas separáveis no frame atual? | Máscara separada não comprova identidade anterior |
| Recorte contaminado por vizinho/grama | É possível isolar melhor o tronco ou corpo? | Não acrescenta exemplos novos à classe de uniforme |
| Pé mal localizado | A silhueta suporta uma estimativa de contato mais estável? | Pé oculto pode continuar indisponível |
| Detecção fundida ou contorno instável | O refinamento confirma a proposta leve? | Não deve criar um atleta duplicado |

Será preciso definir prioridade, limite de chamadas, tempo máximo e comportamento sob falha. Um acionamento em região menor deve ser medido: recortar o input não garante redução proporcional se o modelo o redimensiona para uma resolução interna fixa.

Se houver fila de processamento, a resposta só pode valer quando estiver disponível. Não se pode usar um SAM que terminou depois para afirmar que a previsão antiga já conhecia o resultado. No processamento offline, esperar pelo frame atual é uma alternativa possível, mas seu custo deve entrar no tempo total.

Quando SAM falhar ou não resolver a associação, a resposta deve poder permanecer desconhecida. Não escolher silenciosamente a equipe do slot ATD ou a classe mais frequente para aumentar cobertura.

**Perguntas aos mentores:** quais gatilhos têm maior benefício esperado? Qual orçamento de chamadas garante o tempo alvo e como priorizar casos quando o orçamento acaba?

## 4. Identidade: o risco principal da aceleração temporal

A política aprovada usa quatro instantes em 0,6 s, mas exige continuidade durante os 16 frames do intervalo. Diminuir a frequência da rede de detecção não autoriza ignorar o que ocorre entre as quatro amostras.

O acompanhamento deve usar evidência do frame corrente, distinguir observação de previsão e interromper a identidade quando houver ambiguidade. Mesmo atletas do mesmo time podem se cruzar; concordância de cor não prova que são a mesma pessoa. Também não basta que o número de slot do ATD permaneça igual.

Cenários que precisam de tratamento explícito:

- Cruzamento de pessoas com o mesmo uniforme e oclusão por árbitro ou pessoa externa.
- Saída e retorno à imagem, entrada de substitutos e reutilização de slots ATD.
- Mudança de câmera, corte, início do segundo tempo e homografia inválida.
- Duplicação/fusão de detecções e conflito entre estimativa visual e geometria do tracking.
- Lacuna em frame que não seria amostrado para o uniforme: continua sendo relevante para identidade.

Uma caixa propagada por movimento não deve aparecer como detecção independente certificada. Uma posição prevista não deve confirmar controle individual como se fosse observada. A proveniência precisa indicar detector, acompanhamento, refinamento SAM, disponibilidade e motivo de abstenção.

**Perguntas aos mentores:** quais sinais são suficientes para manter uma identidade? Como medir trocas, fragmentação e falsas recuperações sem uma anotação integral da partida?

## 5. Recorte de treino versus recorte produzido pelo novo detector

O modelo v4 foi construído com recortes revisados. O caminho automático produz recortes a partir de detecção/segmentação e pode mudar a proporção de cabeça, tronco, braços, pernas e fundo. O descritor usa uma região relativa à caixa; uma nova geometria de recorte pode deslocar essa região mesmo quando a pessoa foi localizada corretamente.

Há somente um exemplar de Outro. Os goleiros são pouco representados na avaliação final: dois casos de goleiro Barcelona e um de goleiro Athletic. Na configuração escolhida, foram um acerto e uma abstenção para o primeiro e uma abstenção para o segundo. Isso não sustenta uma alegação de reconhecimento robusto dos dois goleiros.

O caso 169 usa recorte parcial da parte superior do corpo; a mesma imagem foi usada nas quatro configurações, mas ele não é equivalente a um recorte de corpo inteiro. Sombra forte também não foi bem representada.

**Perguntas aos mentores:** devemos manter o classificador HSV inicialmente e avaliar separadamente a qualidade do recorte? Que diversidade mínima de iluminação, goleiros e exemplos negativos é necessária antes de mudar o modelo de uniformes?

## 6. Limites estatísticos e prevenção de vazamento

A comparação temporal teve 32 sequências propostas, 31 rótulos visuais, três exclusões solicitadas pelo usuário e **29 sequências avaliadas em apenas três grupos temporais**. Não são 29 cenas independentes, muito menos milhares de amostras independentes por contar cada frame.

A janela de 0,6 s apresentou 26 acertos, três abstenções e zero erros aceitos. A acurácia dos aceitos foi 100% nessa amostra pequena. O método foi escolhido depois de consultar a comparação; o desempenho da política escolhida precisa ser confirmado em outros trechos.

As exclusões 173/174/175 ocorreram após a rotulagem. Elas são documentadas e uniformes entre as configurações, mas tornam a seleção do subconjunto exploratória. Excluir esses casos específicos não confirma que todos os outros sejam jogo em andamento. O grupo 160000 contém paralisação, que pode ser útil para reconhecer uniforme sem provar posse.

Parte de testes antigos foi promovida a treino em versões anteriores com autorização. A separação atual confere exemplos efetivamente treinados e margens temporais, mas isso não torna o mesmo jogo uma avaliação de generalização entre partidas. A janela pública de posse também já foi consultada repetidamente e coincide com uma região usada no desenvolvimento visual.

Proposta de protocolo para discussão:

1. Definir os grupos de desenvolvimento e avaliação antes de consultar o resultado do novo sistema.
2. Manter uma pessoa/sequência e seus frames no mesmo conjunto; evitar janelas sobrepostas entre conjuntos.
3. Congelar pesos, parâmetros, gatilhos SAM e critérios de exclusão antes da avaliação reservada.
4. Salvar previsões antes dos rótulos humanos de teste e manter o gabarito de posse fora da inferência.
5. Não incorporar novos rótulos ao treino sem autorização; preservar a avaliação histórica.
6. Comparar rapidez e qualidade nos mesmos casos, com grupos temporais como unidade de análise da incerteza.

**Perguntas aos mentores:** qual conjunto mínimo permite decidir entre as arquiteturas? Que trechos devem ser reservados agora e que tipo de intervalo de confiança faz sentido com poucos grupos?

## 7. Acurácia de posse: o que falta demonstrar

O resultado de **73,27%** pertence à execução histórica de contexto 25 + goleiro em 120 s do gabarito público. Os **89,7%** pertencem à classificação de uniformes em 29 sequências. Não são a mesma tarefa nem podem ser comparados como evolução de acurácia de posse.

A nova medição de posse com a política de 0,6 s foi iniciada e pausada; não há novo número de posse. O detector leve e o SAM seletivo tampouco foram implementados. Melhorar detecção, aumentar cobertura de equipes ou reduzir Incerto não garante melhora de posse: também pode introduzir atribuições erradas mais confiantes.

O gabarito oficial público permite avaliar H/A/D/X, macroF1 e reconstrução dos períodos de posse. Ele não fornece sozinho a verdade de identidade, máscara, pés, controle individual ou qualidade do recorte. Esses aspectos exigem revisão visual ou anotações próprias, separadas da avaliação de posse.

**Perguntas aos mentores:** qual perda de cobertura ou aumento de abstenção é aceitável para reduzir trocas de equipe? Como ponderar macroF1 e spellF1, além da acurácia, na decisão de adotar a arquitetura híbrida?

## 8. Plano de validação a discutir antes de escalar

| Camada | Verificações propostas | Evidência de sucesso esperada |
|---|---|---|
| Detecção | Pessoas pequenas, bordas, oclusão e cenas difíceis | Recall e falsos positivos por cenário, com revisão humana |
| Identidade | Cruzamentos, perdas e retorno à imagem | Trocas, fragmentação e falsas associações por grupo |
| Geometria | Pés/homografia e associação com ATD | Erro de posição e taxa de abstenção, sem duplicações |
| Uniforme | Classes, confiança e recorte | Acertos, abstenções e erros aceitos por categoria |
| SAM seletivo | Custo, gatilhos e utilidade | Fração acionada, tempo e quantos casos resolveu/piorou |
| Posse | Mesmo trecho, mesmo gabarito | Acurácia, macroF1, spellF1 e casos alterados |
| Operação | Execução contínua e falhas | Tempo total, throughput, memória, disco e retomada |

Começar com trechos curtos contendo jogo, paralisação, goleiros, cruzamentos e sombra; depois executar os dois minutos públicos completos; só então escalar a uma partida. O teste curto não deve escolher parâmetros pelo melhor score público. Metas numéricas de qualidade ainda precisam ser acordadas; o alvo operacional inicial é até seis horas para 90 minutos, não tempo real.

O acompanhamento de cada comparação deve indicar o que mudou: detector, resolução, frequência, modelo de uniforme, política temporal, recursos visuais de controle e regra de posse. Alterar tudo simultaneamente impede atribuir a causa do resultado.

## 9. Cache, retomada e custo de repetir experimentos

A análise longa atual salva observações e máscaras, mas não tem retomada validada do estado completo de identidade e memória temporal. Reiniciar do último número de frame sem restaurar esses estados pode mudar a previsão; unir arquivos parciais não equivale a uma execução causal contínua.

O fluxo futuro deve poder reutilizar detecções/máscaras e descritivos, permitindo alterar a posse sem repetir a rede visual. A validade do cache precisa depender de fonte, timeline, frame, modelo, pesos, parâmetros e versão do processamento relevante. Um hash isolado não prova que uma identidade declarada estava correta.

É necessário distinguir cache bruto de visão de resultado final de posse; rejeitar registros incompletos/incompatíveis; controlar o pico de memória; e incluir gravação em disco no benchmark. Vídeos, pesos e artefatos derivados devem continuar fora do Git conforme as regras do projeto.

A reprodução também tem uma limitação de ambiente: as verificações atuais foram feitas em macOS e alguns geradores de pranchas usam a fonte Arial por caminho do sistema. Linux/Windows ainda precisam de verificação de fontes e dependências. O modelo local de uniformes e os dados da partida não acompanham o clone; os hashes publicados permitem identificar os artefatos, mas não os tornam disponíveis aos mentores.

**Perguntas aos mentores:** vale priorizar cache/replay e retomada antes de outro modelo? Qual estado mínimo precisa ser persistido para garantir equivalência da retomada?

## 10. Pauta sugerida para a conversa

1. Confirmar o objetivo: medir posse com qualidade conhecida e processar uma partida offline em algumas horas.
2. Escolher uma família de detector e o tipo de geometria que ela deverá fornecer.
3. Definir a evidência necessária para manter identidade e quando abster-se.
4. Definir quais dúvidas justificam SAM e qual seu orçamento máximo.
5. Fixar um conjunto reservado e critérios de adoção para qualidade e desempenho.
6. Priorizar uma primeira entrega pequena, reproduzível e com comparação justa.

**Limite deste pacote:** os 322 testes aprovados e os resultados locais sustentam comportamentos específicos, não certificam robustez nos 90 minutos. O código atual foi publicado para permitir revisão; a arquitetura híbrida está deliberadamente descrita como proposta, aguardando essa conversa.
