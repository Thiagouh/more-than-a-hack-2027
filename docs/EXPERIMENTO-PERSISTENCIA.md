# Experimento pareado: persistência de aquisição 3 × 5

Execução sintética local em 14/11/2026. Relatório visual: `outputs/persistence-3-vs-5-final-v2/report.html`; resultados detalhados, entradas sintéticas e proveniência ficam no mesmo diretório ignorado pelo Git. Nenhum dado real foi utilizado.

## Protocolo

Hipótese: cinco amostras consecutivas podem filtrar confirmações breves espúrias, mas perder controles legítimos curtos e atrasar a confirmação. Não declarar vencedor pelo número de eventos ou blocos.

Cada um dos 20 casos foi executado exatamente duas vezes: aquisição em 3 ou 5 frames, perda fixa em 3, fps 25 e velocidade relativa máxima 5 m/s. As configurações são iguais nos demais parâmetros. Os hashes SHA-256 dos envelopes de entrada foram comparados entre as versões. As expectativas foram gravadas em protocol.json antes de executar cada experimento; os casos são construídos, não uma amostra independente da partida.

A configuração agora aceita `acquisition_persistence_frames` e `loss_persistence_frames`. Quando omitidos, ambos herdam `persistence_frames` (ou seu equivalente efetivo no modo duração). Overrides por papel são contagens explícitas de frames na frequência da entrada. O padrão global continua 3. Os relatórios de qualidade distinguem o atraso mínimo de aquisição e perda; disponibilidade adicional pode aumentá-lo.

## Resultado por caso

A expectativa é a sequência de aquisições de controle, não uma anotação de posse final por frame. Os casos curtos têm uma amostra inicial de aquecimento antes das evidências elegíveis. Tempos e rastros individuais estão no HTML e nos arquivos comparison.json.

| Caso | Esperado | 3 frames | 5 frames | Frames H/A/X diferentes |
|---|---|---|---|---|
| controle_H_sustentado | H | H | H | 2 |
| controle_A_sustentado | A | A | A | 2 |
| goleiro_H_controle | H | H | H | 2 |
| goleiro_A_controle | A | A | A | 2 |
| roubada_sustentada | H → A | H → A | H → A | 4 |
| roubada_curta_4_evidencias | H → A → H | H → A → H | H → H | 13 |
| controle_real_H_3_evidencias | H | H | nenhuma | 1 |
| controle_real_H_4_evidencias | H | H | nenhuma | 2 |
| controle_real_H_5_evidencias | H | H | H | 2 |
| controle_real_A_3_evidencias | A | A | nenhuma | 1 |
| proximidade_sem_controle_3_evidencias | nenhuma | H | nenhuma | 1 |
| conducao_compativel | H | H | H | 2 |
| passe_sem_controle | nenhuma | nenhuma | nenhuma | 0 |
| disputa_sem_vencedor | nenhuma | nenhuma | nenhuma | 0 |
| duplicatas | nenhuma | nenhuma | nenhuma | 0 |
| identificador_novo_todo_frame | nenhuma | nenhuma | nenhuma | 0 |
| continuidade_entre_companheiros | H | H | H | 2 |
| oclusao_e_reaquisicao | H → H | H → H | H → H | 2 |
| afastamento_sustentado | H | H | H | 2 |
| evidencia_atrasada | H | H | H | 0 |

## Mesmo tracking composto, sintético

Os mesmos 20 casos foram concatenados, com descontinuidades explícitas entre eles, em 388 frames (15,52 segundos). A entrada foi novamente processada duas vezes e seus hashes conferem. Isso é uma verificação de integração/continuidade, não uma avaliação de jogo nem evidência estatística adicional.

| Medida | 3 | 5 |
|---|---:|---:|
| Aquisições | 20 | 15 |
| Perdas | 4 | 3 |
| Disputas | 1 | 1 |
| Toques | 1 | 1 |
| Frames H | 160 | 136 |
| Frames A | 77 | 61 |
| Frames X | 151 | 191 |

As saídas diferem em 40 de 388 frames (10,31%). Esse percentual mede divergência, não erro. O número de perdas muda mesmo com limiar fixo porque a aquisição adversária curta deixou de existir como predecessora. A redução de cinco aquisições corresponde a uma confirmação falsa por hipótese e quatro controles legítimos por construção; não deve ser extrapolada para prevalências reais.

## Análise

- **Controles sustentados e goleiros H/A:** ambos detectam; no caso estável, o instante estimado é o frame 1. A confirmação acontece no frame 3 ou 5: atraso mínimo de 80 ou 160 ms desde a primeira evidência, com aumento de 80 ms.
- **Controles legítimos com 3/4 evidências:** cinco deixa de detectá-los. O resultado mostra o custo deliberado de maior persistência, não falha no código.
- **Roubada curta:** jogadores aproximam-se/afastam-se continuamente a 10 m/s; o controle A tem quatro amostras de movimento compatível, nos frames 25–28. Três confirma A no frame 27 e depois H no 46; cinco não confirma A e volta a confirmar H no 48. O consumidor produz H → X → H nesse trecho, não conclui que H manteve posse. A série de aquisições é H → A → H contra H → H.
- **Proximidade sem controle:** cinco evita a aquisição espúria de três. Esse caso tem envelopes idênticos ao controle curto legítimo: a diferença está na hipótese esportiva que o tracking não observa. Não é possível resolver ambos apenas mudando a persistência.
- **Passes, disputa, duplicatas e IDs novos:** os dois perfis evitam aquisições indevidas nesses casos específicos. Isso não prova cobertura de todas as variantes.
- **Atraso de produtor:** uma evidência disponível apenas no frame 15 faz ambos os perfis chegarem ao consumidor no frame 15. Confirmação e disponibilidade não são a mesma medida.
- **Oclusão:** detector exige nova sequência após a lacuna; o consumidor event-only conserva o último time até receber perda ou aquisição contrária. Continuidade H/A nesse intervalo é limitação do protótipo, não confirmação observada.

Não escolheria cinco como superior com estes dados. A escolha troca sensibilidade a controles rápidos por maior estabilidade. Antes de alterar a decisão final, precisamos de referência independente para distinguir controle real de aproximação. Uma alternativa futura é preservar candidatos de três e exigir evidência adicional para uma troca final de posse; isso exige definir posse e integrar mais fontes na frente 4, não apenas aumentar um contador.

## Frente 4 provisória

Após `git fetch origin`, não houve atualização no remoto configurado; a única branch remota de trabalho continua origin/main e não contém implementação da frente 4. Foi adicionado `src.team_possession`, um consumidor experimental comum aos dois perfis:

- começa em X e recebe eventos somente em `available_at_frame`, sem reescrever o passado;
- aquisição define H/A; perda da aquisição ativa define X;
- perdas de aquisições já substituídas e aquisições antigas entregues tarde não desfazem decisões mais novas;
- mudança de segmento reinicia X; eventos atrasados de segmento anterior são descartados;
- toque e disputa não impõem vencedor; não produz D nem usa gabarito.

A política mantém o último estado entre eventos. Não tem observações de visibilidade, arbitragem, bola fora, definição completa de posse ou estimativa final retrospectiva. Portanto, não é uma implementação final nem uma submissão recomendada da frente 4. Os CSVs de saída passaram pela validação estrutural oficial para seu próprio tamanho sintético; isso não certifica tamanho oficial nem precisão esportiva.

## Tracking real: execução posterior concluída

Após receber o novo link autorizado do Drive, foram baixados os dois arquivos tactical ATD para data/tactical/, ignorados pelo Git. A comparação real de 3/5 foi concluída nos 199.319 frames, sem gabarito. A grafia “Atletic Club” nos metadados passou a ser aceita como variante explícita de “Athletic Club”, com regressão sintética; nenhum vínculo de time foi alterado.

Resultados, análise e recortes de divergência permanecem locais em `outputs/persistence-real-3-vs-5-v1/same-tracking-real/ANALISE.md` e `report.html`. A análise distingue evidência observada de estados mantidos durante ausência de bola; isso limita a interpretação da comparação. Vídeo e cores reais dos goleiros continuam sem inspeção nesta execução.

Reproduzir a análise dos resultados salvos:

```sh
.venv/bin/python -m scripts.analyze_persistence outputs/persistence-real-3-vs-5-v1 --tracking data/tactical/BAR-ATH_tactical_atd.txt
```

A comparação está preparada para repetir as mesmas duas configurações no tracking real, verificando identidade da entrada e registrando diferenças sem atribuir melhora:

```sh
.venv/bin/python -m scripts.compare_persistence --out outputs/persistencia-real --data-dir /caminho/que/contem/tactical
```

Para um manifesto de tracking padronizado, substituir `--data-dir` por `--input /caminho/manifest.json`. O destino deve ser novo. Saídas reais e derivados permanecem locais e ignorados pelo Git.

Reproduzir apenas a execução sintética:

```sh
.venv/bin/python -m scripts.compare_persistence --out outputs/persistencia-repeticao
.venv/bin/python -m unittest discover -s tests -v
```

Verificação desta entrega: 76 testes passaram, incluindo a suíte anterior, configuração independente, comparação pareada, atrasos, prioridade temporal e consumidor provisório. Os hashes dos três arquivos do avaliador oficial continuam preservados. Nenhum score da partida foi calculado, nem parâmetro escolhido pelo gabarito público.
