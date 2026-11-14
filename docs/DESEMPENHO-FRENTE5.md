# Desempenho computacional — frente 5

Medições sintéticas em **04/11/2026**, sem assets ou gabarito da partida. Não medem acerto esportivo, integração real nem generalização.

## Método

199.319 frames, 22 jogadores por frame, 17.556 IDs distintos (todos os rastros mudam a cada 250 frames). Detector e pipeline rodam em processos separados. O modo detector alimenta objetos incrementalmente, sem reter features/eventos; o pipeline gera CSVs sintéticos, faz parsing, usa SQLite e exporta todos os artefatos. A geração de CSVs é medida separadamente. Inputs grandes são temporários e removidos ao final.

Churn alterna observed/estimated a cada frame: interrompe persistência e não produz eventos. Events mantém bola observada com deslocamentos e pausas, gerando aquisição, perda e toque. A simulação computacional não se propõe a representar uma partida real.

Pico RSS = máximo do processo, incluindo imports e preparação; não equivale apenas à memória do detector. Armazenamento temporário = tamanho observado do SQLite e journal imediatamente antes de fechar, não um monitor contínuo do pico em disco. MB decimais (1 MB = 1.000.000 bytes). Tempos de uma execução nesta máquina; não usar como benchmark comparativo entre algoritmos ou garantia de recursos.

## Resultados

| Padrão/snapshot | Modo | Tempo (s) | Pico RSS (MB) | Eventos | SQLite/journal ao fechar (MB) |
|---|---|---:|---:|---:|---:|
| Alternância/churn (medição inicial) | detector | 48.40 | 30.36 | 0 | 0.00 |
| Alternância/churn (medição inicial) | pipeline | 66.20 | 40.45 | 0 | 3.56 |
| Eventos (dependências refinadas) | detector | 84.92 | 30.70 | 19932 | 0.00 |
| Eventos (dependências refinadas) | pipeline | 108.80 | 42.11 | 19932 | 84.17 |

Ambiente local: Python 3.14.6, Darwin, arm64. O custo maior da versão com dependências refinadas é registrado, sem apresentar apenas a medição inicial mais rápida. Cada pipeline inclui hashes do snapshot de código efetivamente medido em manifest.json; o HEAD sozinho não identifica mudanças ainda não versionadas.

A geração dos CSVs levou cerca de 12 s por cenário, fora do tempo do pipeline. O cenário de eventos refinado exportou aproximadamente 105 MB; esses arquivos e o SQLite temporário não entram no Git.

## Aceite final do pipeline

Após os últimos checks semânticos do contrato, foi executado novamente o pipeline completo em `outputs/frente5/escala-v2-entrega/`: **106.59 s**, pico RSS de **41.17 MB**, **19,932 eventos sintéticos**, SQLite/journal observado de **84.17 MB** e artefatos de **105.38 MB**. Geração de CSVs: 11.79 s, separadamente. Os hashes do código desse manifesto foram comparados com o estado entregue: 0 divergências. Inputs sintéticos temporários foram removidos ao concluir.

## Reprodução

```sh
.venv/bin/python scripts/benchmark_frente5.py --out outputs/frente5/nova-escala-churn
.venv/bin/python scripts/benchmark_frente5.py --pattern events --out outputs/frente5/nova-escala-eventos
```

Para medir somente o pipeline, acrescentar `--mode pipeline`. Escolher destinos novos. Relatórios completos estão em outputs/frente5/escala-v2/, escala-v2-eventos/ e escala-v2-final/. A medição adicional de aceite dos validadores finais foi registrada em escala-v2-entrega/.

## Limites

O registro padrão de IDs na API Detector é O(IDs); o pipeline usa SQLite e cache de até 2.048 IDs. Pistas e seu índice são O(pistas). Detect() acumula as features e eventos deliberadamente. Eventos/origens no pipeline ficam em disco, com leitura e exportação incremental. A visão geral usa 980 bins de origem e arquivo SVG incremental; marcas de eventos ainda crescem em quantidade/tamanho no arquivo e podem se sobrepor.

O matching externo usa janelas e componentes; um componente muito denso ainda pode exigir muitos candidatos e processamento. Não foi medido em dados externos nesta rodada. Esses limites são distintos de qualidade dos eventos.
