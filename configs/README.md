# Configurações dos experimentos

Guarde aqui os parâmetros das implementações da equipe: por exemplo, limiares geométricos,
pesos das evidências e custos de transição. Não há configuração fictícia criada: o primeiro arquivo
deve acompanhar o primeiro módulo que realmente o utiliza.

Cada execução deve registrar em `outputs/` a configuração e o commit utilizados. Não ajustar
parâmetros nos cinco minutos públicos nem introduzir constantes específicas para frames do jogo.

Os perfis `frente5_v1.json` e `frente5_dois_frames.json` preservam os nomes dos experimentos originais. Ambos declaram persistência em frames a 25 Hz e agora produzem eventos **v2**. Para outras frequências ou persistência em duração, consulte [o guia](../docs/FRENTE5-EVENTOS.md).
