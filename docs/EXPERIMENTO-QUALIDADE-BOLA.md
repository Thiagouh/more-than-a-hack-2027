# Experimento: qualidade física da bola antes da posse

Protocolo definido antes de executar esta variante. A referência estável permanece
a Frente 4 original, commit de integração `537b678`.

## Hipótese

Observações de bola com saltos fisicamente incompatíveis podem gerar candidatos
ou paralisações falsas. Retirá-las deve impedir esse uso, sem criar novas posições.
Uma bola que desaparece logo após uma observação além da linha pode fornecer um
gatilho explícito de D que a regra de três frames fora não consegue produzir.

## Regras fixadas

- Apenas coordenadas finitas são observações elegíveis.
- Deslocamento entre frames consecutivos acima de 40 m/s invalida os dois extremos
  e dois frames adjacentes de cada lado, seguindo as constantes já propostas pela
  Frente 2. Observações mais de 5 m fora do campo também são retiradas.
- Nenhuma lacuna é interpolada ou preenchida com previsão.
- Se a última observação válida antes de uma lacuna estiver mais de 0,5 m além da
  linha, o primeiro frame ausente pode abrir D. O restante da política de D e de
  retomada permanece inalterado.
- A opção deve ser explícita (`--ball-quality`); a referência sem opção deve
  reproduzir o CSV estável, byte a byte.

Não há treinamento, calibração no jogo ou busca de parâmetros. As constantes de
limpeza vêm da Frente 2 e a margem de saída vem da Frente 4. A geração não abre
gabaritos. Qualquer checagem pública posterior é de integração; não selecionará
limiares, janelas ou exclusões.

## Verificação antes de adoção

Testar saltos, chutes plausíveis, lacunas dentro do campo, saída seguida de lacuna,
coordenadas parciais e não mutação dos inputs. Confirmar que observações mantidas
são subconjunto exato das originais, que faltas de observação não abrem D sozinhas,
que a timeline é preservada e que duas gerações independentes coincidem.

A regra física pode retirar um chute real acima do limiar. A posição fora das
linhas também pode estar errada. Essas hipóteses serão explicitadas no relatório;
não haverá revisão manual ou correção por frame.

## Verificação externa predefinida

Usar os três primeiros IDs, em ordem numérica, do repositório oficial SkillCorner
Open Data: **1874553, 1886347 e 1899585**. Fonte congelada no commit
`4340d274572876239c154c90bc507a9b3250a656`, licença MIT. A escolha é anterior à
leitura dos rótulos e aos resultados desses jogos.

O tracking é de transmissão, a 10 fps e com coordenadas centradas. Converter para
coordenadas positivas com as dimensões de cada campo. A dilatação equivale a dois
frames em 25 fps (0,08 s), reescalada ao fps com arredondamento e mínimo de um frame.
Essa conversão não altera a execução de 25 fps da competição.

Comparar cobertura e erros dos candidatos geométricos somente nos frames em que
o fornecedor declara equipe em posse, antes e depois do filtro. Esses rótulos não
são o gabarito da competição e não validam bola morta; não preencher lacunas de
rótulos como se fossem verdade conhecida. Não treinar nem ajustar limiares nesta
verificação. O teste externo é separado das contagens automáticas do jogo alvo.

Fonte: [SkillCorner Open Data](https://github.com/SkillCorner/opendata).
