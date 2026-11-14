# Análise das contribuições disponíveis

Revisão em 14/11/2026, após `git fetch origin`. Base: `954afaf`; main e origin/main estavam sincronizadas. O remoto configurado é o fork Thiagouh/more-than-a-hack-2027. Esta análise cobre seu histórico e código disponível, não trabalhos em outros forks, branches não publicadas ou mensagens da equipe.

## Autoria registrada

- **jcnunezmoreno — quatro commits (e47adea, b429693, f00bc32, 3de6b60):** entrega inicial da competição, documentação, avaliador, validação de submissões, baselines, leitores e notebooks; os três commits seguintes corrigem documentação/citação. É a infraestrutura de referência, não evidência de uma solução nova da equipe.
- **Thiago Lorhan — três commits (68e1090, cfdf63c, 954afaf):** organização em docs/notebooks, preparação local de assets, scripts de conferência e apresentação pública em português/inglês. O documento ORGANIZACAO.md registra verificações anteriores; elas não foram reexecutadas com dados reais nesta revisão.
- **Demais integrantes:** nenhum outro autor ou commit apareceu nas referências disponíveis após o fetch. Não é possível avaliar contribuição individual não registrada nesse histórico. Arquivos que estavam sem commit não têm autoria individual demonstrável pelo Git.

## Avaliação técnica

1. **Avaliador original:** deve permanecer como referência. A suíte atual verifica os hashes de possession_metrics.py, run_evaluation.py e validate_submission.py, além de comportamentos de X/D. Os testes passaram; isso não constitui uma auditoria exaustiva de todas as métricas.
2. **Baseline original:** starter/baseline.py atribui o time do jogador mais próximo e suaviza por janela centrada. É útil para comparação reproduzível, mas proximidade não demonstra controle e a janela usa amostras futuras. Para integração causal é necessário explicitar disponibilidade/atraso. A ausência de observação também não demonstra bola morta. São limites do baseline, não defeitos no contrato oficial.
3. **Organização de Thiago:** melhora orientação e separação entre assets, código e saídas. A execução sintética atual passou. A validação anterior de arquivos reais está documentada, mas não foi repetida aqui.
4. **Frente 5 pendente de commit:** há detector de eventos candidatos v2, contratos/esquemas, proveniência, auditoria, cache, reprodução temporal, comparação de CSVs e visualização. A suíte de 60 testes sintéticos passou após remover um caractere indevido no início de evaluation/desenhar_posses.py. Ainda faltam consumidor real da frente 4 e avaliação esportiva com dados e referência apropriados; não há evidência de ganho de pontuação.

## Próxima integração

Priorizar o consumidor da frente 4 e testar a entrega dos eventos com seus atrasos e predecessores. Solicitar links de branches/PRs dos demais integrantes para uma revisão por mudança. Não atribuir resultados nem qualidade individual com base apenas em quantidade de commits.

## Verificação desta revisão

- Fetch do remoto concluído; nenhuma atualização recebida.
- `.venv/bin/python -m unittest discover -s tests -v`: 60 testes, OK.
- `git diff --check`: sem problemas.
- Nenhum scoring da partida, consulta de gabarito ou publicação remota executados nesta revisão.
