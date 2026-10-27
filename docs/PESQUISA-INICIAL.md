# Pesquisa inicial — 26/10/2026

Referências e observações técnicas para entender o material fornecido. Não define arquitetura, responsáveis, cronograma interno ou divisão de trabalho. Essas decisões ficam para a próxima reunião.

## O que a pesquisa indica

1. **Priorizar posses completas.** O ranking usa spellF1, que compara intervalos de posse, com macroF1 como primeiro desempate. Portanto, reduzir alternâncias falsas e acertar mudanças de controle é central. A referência publicada para o jogo completo é spellF1 0,2433; esse valor não é diretamente comparável a uma nota obtida apenas nos cinco minutos públicos. [Avaliação oficial](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/EVALUATION.md).
2. **Bola ausente não significa bola parada.** O baseline confunde essas situações e ignora eventos e estrutura temporal. É uma oportunidade concreta de melhoria: `starter/baseline.py:109–175`, especialmente a conversão de observações ausentes para D em `smooth`. A ausência da bola precisa de tratamento explícito de incerteza.
3. **Posse exige controle.** Bola solta ou disputada mantém a equipe que tinha o controle anterior até haver controle adversário; proximidade ou um desvio isolado não bastam. [Definições oficiais](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/CHALLENGE.md).
4. **Smart Tagging é evidência ruidosa.** Seus intervalos são fases, não posses exatas; tags podem se sobrepor ou não indicar equipe. Não copiar diretamente esses intervalos como resposta. [Dados oficiais](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/DATA.md).
5. **Uma abordagem geométrica tem fundamento, mas precisa de adaptação.** Vidal-Codina et al. usam zonas de controle e mudanças de movimento da bola para inferir posse e eventos. O artigo oferece ideias para features, não comprova desempenho neste desafio. Sua lógica de ausência de tracking como sinal de bola morta não deve ser transplantada para este ATD com oclusões. Leitura inicial: resumo e seções de posse, controle e reposições. [Artigo original](https://link.springer.com/article/10.1007/s12283-022-00381-6).
6. **Decodificação temporal é uma hipótese de estudo.** Viterbi escolhe uma sequência de estados por programação dinâmica em um grafo temporal. Podemos atribuir custos a H/A/D em cada frame e às mudanças entre estados, buscando o caminho de menor custo. É uma proposta técnica a validar, sem garantia de superar o baseline. [Berkeley: Viterbi](https://inst.eecs.berkeley.edu/~cs188/textbook/hmms/viterbi.html), [interpretação como caminho mínimo](https://cim.mcgill.ca/~latorres/Viterbi/va_alg.htm).
7. **Dados externos são uma frente inicial de pesquisa.** O sample-data da Metrica fornece tracking e eventos sincronizados de outros jogos. É candidato para desenvolvimento e validação, após conferir termos e compatibilidade das definições. Eventos não constituem automaticamente um ground truth completo de posse por frame. [Fonte original](https://github.com/metrica-sports/sample-data).

## Regras atuais: diferença importante do checkout

O `RULES.md` local, no commit `3de6b60`, ainda contém uma permissão genérica para pequenas quantidades de rótulos manuais. **A versão oficial atual é mais restritiva:** proíbe rótulos e correções manuais nesta partida, em qualquer quantidade, inclusive para treinamento ou validação. Os cinco minutos públicos servem para verificar o pipeline, não para treinar, ajustar parâmetros ou escolher modelos. Também não se pode buscar dados humanos sobre esta mesma partida em provedores externos. Outros jogos podem ser usados. Inspeção visual para compreender e depurar é permitida, sem convertê-la em correções ou constantes específicas para frames. [Regras oficiais atuais](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md), [FAQ atual](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/FAQ.md), consultadas em 26/10/2026.

Dados da competição e derivados que os reconstruam não devem ser publicados. Na organização posterior à pesquisa, os ZIPs foram movidos para `data/archives/`; `data/` e arquivos ZIP estão ignorados pelo Git.

## Estado local verificado

Inspeção dos membros dos nove ZIPs e leitura sequencial dos arquivos tabulares, sem extrair ou modificar o dataset:

| Item | Resultado em 26/10 |
|---|---|
| `data/` | Após a organização: sete assets tabulares extraídos e dois vídeos nos caminhos dos loaders; arquivos originais ZIP preservados em `archives/` |
| Tracking tactical | 199.319 registros contíguos, IDs 1 a 199319; bola presente em 113.488 registros, 56,94% do vídeo completo |
| Tracking broadcast | 164.488 registros contíguos, IDs 1 a 164488; bola presente em 50,53% |
| Homografia tactical | 199.319 linhas em ordem, todas marcadas como válidas |
| Homografia broadcast | 164.488 linhas em ordem, 65,81% marcadas como válidas |
| Smart Tagging | 638 instâncias; 36 tags Team=N/A; 59 SET PIECES |
| Slots de tracking | 46 tactical / 51 broadcast; não são identidades estáveis de jogadores |
| Vídeos | Localizados posteriormente na raiz e movidos para `data/`; tactical declara 199.319 frames; broadcast declara 164.487, um a menos que seu tracking |
| Ambiente | `.venv` criado com numpy e pandas; dependências básicas fixadas em `requirements-lock.txt`; ferramentas opcionais dos notebooks não instaladas |

As porcentagens acima são do vídeo completo e não medem acurácia. A pesquisa inicial não executou o pipeline completo. Na organização posterior, o ambiente foi isolado em `.venv` e os quatro baselines foram regenerados com conteúdo idêntico aos CSVs originais. Não houve treino ou ajuste de parâmetros. Consulte [ORGANIZACAO.md](ORGANIZACAO.md) para a validação final.

## Possibilidades de avaliação a discutir

Separar desenvolvimento e validação por **partida**, sempre que houver jogos suficientes. Frames vizinhos de um mesmo trecho não devem cair aleatoriamente em treino e validação. Quando houver só uma partida externa, a evidência será mais limitada e isso deve ser registrado.

Antes de usar os jogos externos para decidir o modelo, conferir a definição de posse e como o estado D foi obtido. Se só houver eventos esparsos ou pseudo-rótulos, medir e nomear o resultado como proxy; não apresentar como spellF1 confiável de posse completa. Se não houver referência adequada, manter uma solução simples com critérios gerais, testes sintéticos e diagnóstico sem rótulos; registrar a incerteza, sem usar o GT público como substituto.

Experimentos prioritários: baseline oficial; controle com persistência; inclusão de evidência D; decodificação temporal; combinação das partes. Fazer ablações removendo um componente por vez no conjunto externo. Registrar commit, configuração, dados, seed, tempo, spellF1/macroF1 quando a referência permitir e falhas observadas. Contagens de posses, suavidade e concordância com Smart Tagging são diagnósticos, não prova de acurácia.


## Preparação dos dados locais

Cada participante deve obter o pacote autorizado e instalá-lo em seu próprio computador, conforme
[DADOS-LOCAIS.md](DADOS-LOCAIS.md). Nenhum asset é distribuído pelo repositório. Os arquivos que
já existem nesta máquina continuam locais e ignorados pelo Git.
