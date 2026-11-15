# Pacote de revisão — contexto de 25 frames e goleiro

Este pacote acompanha o código, a documentação e os testes no mesmo commit. Permite conferir o resultado final sem copiar o vídeo tactical, pesos de modelo ou ambientes locais.

- [Relatório comparativo](report.md): resultados mistos, ablações, evidências e limitações.
- [Animação de três versões](comparison.mp4): tracking com gabarito público, anterior v4, contexto 25 e contexto/goleiro; 120 segundos, 3.000 frames, 25 fps.
- [Prévia](preview.png), [meio](qa-midpoint.png) e [fim](qa-end.png).
- [Métricas e escopo](assessment.json), [verificações](verification.json), [mudanças por frame](changes.csv) e [testes](tests.log).
- [Proveniência da animação](simulation-provenance.json): hashes das entradas e saídas efetivas. Os caminhos absolutos identificam a execução original e não precisam existir neste checkout para assistir ao MP4.

O contexto isolado reduz X de 534 para 463. A combinação com a revisão do goleiro registra 574 X, acurácia de 73,27% e F1 de segmentos de 0,5524. Não se deve apresentar a combinação como redução global de desconhecidos nem como reconhecimento automático completo.

Os arquivos sob `data/` e os experimentos completos sob `outputs/` permanecem locais. Os comandos e contratos para reprodução estão em [POSSE-COLETIVA](../../docs/POSSE-COLETIVA.md) e [PILOTO-SAM3](../../docs/PILOTO-SAM3.md). A reprodução integral requer as entradas originais e os pacotes revisados citados nesses documentos.
