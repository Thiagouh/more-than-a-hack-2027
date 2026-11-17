# Experimento: controle automático com suporte explícito

Hipótese definida antes da integração: a evidência de controle da Frente 5 pode
reforçar a máquina de estados da Frente 4 sem ampliar artificialmente a duração
observada. Usar a configuração já existente `configs/frente5_v1.json`, sem ajuste
no jogo: controle a até 1,5 m, margem de 1 m, velocidade relativa até 5 m/s e três
frames de persistência.

O adaptador antigo mantém um evento por 12 frames. Nesta variante, a candidatura
do evento só vale de `event_frame` até `confirmed_at_frame`, inclusive. Não
transformar três observações em 12 amostras independentes. As regras de troca da
Frente 4 continuam as mesmas; os eventos não forçam uma troca instantânea.

## Contratos obrigatórios

- O bundle deve ter manifestos e todos os arquivos com hashes válidos, a mesma
  timeline, hashes dos mesmos ATD bruto/metadados e o mesmo kernel/configuração
  do detector. Nenhuma referência ou correção humana entra na integração.
- Aceitar somente aquisições automáticas H/A com força ordinal ao menos média e
  sem flags de qualidade. Força não é probabilidade calibrada.
- Se o filtro físico retirou uma observação de qualquer parte do suporte do
  evento, não reintroduzir essa evidência por outra via.
- Suportes contraditórios entre equipes devem ficar desconhecidos nos frames em
  conflito. A ordem dos arquivos não decide o time.
- D continua sendo decidido pela camada de paralisação, não por estes eventos.
- Sem as novas opções, reproduzir o CSV estável de SHA256 `98e3d571...ccce4df`.

Verificar separadamente geração e integridade dos eventos, limites temporais,
conflitos, flags, rejeição por suporte removido e reprodução completa. A geração
não recebe gabaritos. Uma checagem pública posterior não escolherá duração,
limiares ou força mínima. A variante pode ser regenerada em um comando com
`--include-control-events`; `--control-events` reutiliza um bundle validado.

## Resultado automático e decisão

O bundle completo contém 951 aquisições automáticas de controle. A integração com
filtro físico aceitou 915 e rejeitou 36 por suporte de bola indisponível. São
2.745 frames sustentados, três por aquisição aceita, sem prolongamento artificial.
Isso mudou 13 labels H/A em relação ao filtro físico isolado, sem mudar D ou a
contagem total de períodos previstos. Essas contagens não demonstram quais labels
estão corretos.

Decisão de engenharia, antes de consultar a nova checagem pública: usar somente
o suporte efetivo e íntegro, preservando o piso de força da configuração original
e os critérios de persistência da Frente 4. A referência original permanece
reproduzível sem opções. Não haverá busca por duração de evento ou limiar pelo
score público.
