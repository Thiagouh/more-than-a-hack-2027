# Dados externos — decisão da frente 5

Data: **03/11/2026**, antes do limite de 05/11/2026. Pesquisa de documentação primária e catálogo; **nenhum tracking externo foi baixado e nenhuma validação real foi executada**.

## Recomendação

**Adotar um piloto limitado de eventos IDSSE como próximo experimento externo, mantendo esta entrega tecnicamente aceita por testes sintéticos.** Não tratar posse do fornecedor como verdade independente enquanto seu processo de geração não estiver estabelecido. A recomendação não depende de pontuações do detector.

A consulta de metadados à [API oficial Figshare](https://api.figshare.com/v2/articles/28196177), versão 1, confirmou CC BY 4.0 e 21 arquivos (três por partida). Escolha objetiva: **DFL-MAT-J03WOH**, menor conjunto completo no catálogo:

| Arquivo | ID Figshare | Bytes |
|---|---:|---:|
| Match information | 51643478 | 11.745 |
| Events raw | 51643499 | 635.747 |
| Positions raw observed | 51643520 | 348.945.715 |
| Total | — | 349.593.207 |

O arquivo de posições tem aproximadamente 349 MB e não precisa ser baixado para entregar contrato/detector. Download completo fica adiado para o piloto: primeiro confirmar a semântica e os tempos com os dois arquivos pequenos; depois processar só esse jogo localmente por streaming. Não baixar o pacote completo de sete jogos.

## Fontes e adequação

| Fonte primária | Tracking/frequência | Licença | Referência de eventos/controle |
|---|---|---|---|
| [IDSSE, artigo dos autores](https://pmc.ncbi.nlm.nih.gov/articles/PMC11787359/) e [dataset](https://figshare.com/articles/dataset/An_integrated_dataset_of_spatiotemporal_and_event_data_in_elite_soccer/28196177) | Sete jogos; jogadores e bola, 25 Hz, coordenadas métricas; sem vídeo no pacote | CC BY 4.0 | Eventos anotados por operadores em processo separado do tracking; atributos BallPossession e BallStatus existem, mas sua geração não deve ser presumida independente |
| [Metrica Sample Data](https://github.com/metrica-sports/sample-data) e [metadata jogo 3](https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_3/Sample_Game_3_metadata.xml) | Jogadores/bola, eventos sincronizados; metadata do jogo 3: 25 Hz; eixos 0–1 | README pede atribuição, sem licença padronizada explícita | Útil para leitura e consistência. Independência do processo de anotação não estabelecida pela documentação consultada |
| [SkillCorner Open Data](https://github.com/SkillCorner/opendata) | Dez partidas A-League, jogadores/bola a 10 Hz; detecção/extrapolação explícitas | MIT; atribuir a fonte | Dynamic Events são derivados. Servem como comparação de consistência, não como gabarito independente |

Não confundir 25 fps de Body Pose interpolado no SkillCorner com tracking XY observado a 25 Hz. Não converter 10 Hz em 25 Hz e alegar ganho de resolução temporal.

### Independência e sincronização IDSSE

O [artigo](https://pmc.ncbi.nlm.nih.gov/articles/PMC11787359/) descreve eventos humanos e timestamps com erro, além de sincronização por características como distância e aceleração da bola. Essa sincronização cria dependência justamente de sinais usados pelo detector. Há descrições distintas sobre sincronização em trechos do material; a versão e o arquivo efetivamente usado precisam ser identificados antes do piloto.

Usar os eventos originais para uma análise separada com tolerâncias fixadas; se também houver timestamps sincronizados, reportá-los em separado com a dependência declarada. Nunca ajustar a referência usando a saída do detector. Ausência de informação sobre geração da posse significa **independência desconhecida**, não autorização para chamar BallPossession de verdade.

## Protocolo fechado do piloto

1. Registrar licença, URL/versão, nomes, tamanhos e hashes dos arquivos selecionados; registrar hipótese antes de processar. Não usar a partida do desafio como validação externa.
2. Confirmar mapeamento home/away, dimensões, unidade de velocidade (o fornecedor pode usar km/h), início de períodos e relação timestamp/frame. O detector deriva velocidade das posições métricas; não copiar velocidade em km/h como m/s.
3. Fixar desenvolvimento no primeiro tempo de J03WOH e avaliação no segundo tempo, com reset na fronteira. Isso não é teste entre partidas e não sustenta generalização.
4. Preservar IDs de eventos; extrair apenas passes/chutes com time e instante definidos. Marcar excluídos por falta de semântica/alinhamento e reportar cobertura. Não criar rótulos de controle entre ações.
5. Usar o adaptador externo somente como conversão para contrato v1. Ele ainda não foi implementado: a entrega aceita referência/posições já padronizadas. Antes de implementá-lo, validar os dois XMLs pequenos e sua sincronização; não adivinhar o formato.
6. Congelar parâmetros antes do segundo tempo. Avaliar possíveis toques com mesmo time, associação um a um e tolerâncias 0,2/0,5/1,0 s pelo comando `evaluate-events`. Reportar latência e recall sobre a referência coberta. Predições sem referência não são falsos positivos automaticamente.
7. Não medir aquisição/perda/disputa como acerto sem referência correspondente e independente. Rótulos gerados por regras ou fornecedor automático só medem consistência.

## Se o piloto não for viável

Se faltarem licença clara, tempos compatíveis ou semântica independente, registrar **adiamento da validação real** e motivo. Continuar com cenários sintéticos, oclusões/saltos/duplicatas artificiais, reprodução e contrato com a frente 4. Não usar melhora na janela pública como substituto de validação externa, não criar anotações manuais novas e não alegar desempenho real.

A entrega atual conclui a pesquisa/recomendação e fornece a ferramenta de comparação com referência parcial. Download/adaptação IDSSE e medição real permanecem uma etapa experimental externa explicitamente não executada.

## Atualização de 04/11/2026 — após a revisão

O usuário escolheu concluir correções antes de executar o piloto. **Nenhum dataset foi baixado nesta rodada.** A consulta de catálogo de 03/11 continua sendo evidência registrada no texto, não um snapshot versionado revalidado em 04/11. A recomendação é condicionada à conferência do material efetivo.

Pendências anteriores ao piloto: guardar resposta original de metadados com URL, data, versão, licença, IDs/tamanhos e hash; fixar commit dos repositórios auxiliares; obter e identificar por hash os XMLs pequenos de informação/eventos; confirmar geração de posse e condição dos timestamps. Depois disso, decidir sobre download do tracking escolhido. Esses passos não foram executados aqui.

A ferramenta evaluate-events agora exige contrato v2, IDs únicos e manifesto de timeline. Sua regra foi atualizada: **máxima cardinalidade e mínimo erro absoluto**, com pares publicados e tolerâncias exatas de 0,2/0,5/1,0 s. O texto anterior que descreve associação gulosa é histórico e não rege a nova execução. Erro de localização e atrasos de confirmação/disponibilidade são reportados separadamente. Congelar esta versão e a configuração antes da separação desenvolvimento/avaliação.

Os resultados continuam parciais: eventos do fornecedor não cobrem todos os toques e não validam controle entre ações. Referência derivada/sincronizada por tracking deve ser identificada como tal. B2/B3 da revisão permanecem pendentes: integração com frente 4 e qualidade esportiva não foram medidas.
