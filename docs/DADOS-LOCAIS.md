# Como preparar os dados no seu computador

O GitHub entrega código, documentação e guias. **Os dados da partida não acompanham o clone.**
Cada participante deve usar o acesso concedido pela organização após inscrição e aceite dos termos.
Não enviar o pacote no GitHub, em releases ou em anexos públicos. As [regras oficiais](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md)
proíbem redistribuir os assets e derivados que permitam reconstruí-los.

## 1. Obter o pacote autorizado

Use o link enviado pela organização à equipe. A pasta informada nesta conversa foi
[este Drive](https://drive.google.com/drive/folders/1hRf0Jaah5EXKJe2LFTEITYldpWZRgaD5).
Se sua conta não tiver acesso, utilize o fluxo de inscrição/contato da organização; o repositório
não concede acesso nem redistribui uma cópia. Não é necessário compartilhar credenciais.

## 2. Colocar nos destinos esperados

Não deixe uma pasta externa com o nome do download entre `data/` e as duas subpastas:

```text
data/
├── tactical/
│   ├── README.md                         guia versionado
│   ├── BAR-ATH_tactical.mp4               seu arquivo local
│   ├── BAR-ATH_tactical_atd.xml
│   ├── BAR-ATH_tactical_atd.txt
│   ├── BAR-ATH_tactical_homography.csv
│   └── BAR-ATH_tactical_smart_tagging.xml
└── broadcast/                            entrada opcional
    ├── README.md                         guia versionado
    ├── BAR-ATH_broadcast.mp4
    ├── BAR-ATH_broadcast_atd.xml
    ├── BAR-ATH_broadcast_atd.txt
    └── BAR-ATH_broadcast_homography.csv
```

O tactical é a referência da submissão. O broadcast tem timeline própria; baixá-lo não resolve a
sincronização entre fontes. Os dados usados localmente nesta máquina ocupam cerca de 19,6 GB
sem contar os arquivos ZIP; reserve espaço adicional para downloads e experimentos.

## 3. Se os dados estiverem em outro disco

A variável aponta para a pasta que contém `tactical/` e `broadcast/`:

```bash
export MTAH_DATA_DIR=/caminho/para/meus-dados
```

PowerShell:

```powershell
$env:MTAH_DATA_DIR = "D:\mtah2027"
```

Não é necessário editar `games.py`. Evite gravar no código o caminho pessoal de um colega.

## 4. Conferir antes de começar

Na raiz do projeto, sem instalar bibliotecas adicionais:

```bash
python scripts/verificar_dados.py
python scripts/verificar_dados.py --source all
python scripts/verificar_dados.py --data-dir /outro/disco/meus-dados
```

O primeiro comando exige apenas os cinco arquivos tactical; `--source all` confere os nove.
A verificação só lê a existência e o tamanho dos arquivos: não baixa, corrige nem valida o conteúdo
completo dos vídeos. Depois, os loaders podem conferir alinhamento e formatos.

## O que fica no Git

Somente os arquivos `README.md` de orientação em `data/` e nas subpastas tactical/broadcast.
MP4, XML, TXT de tracking, CSV de homografia, ZIPs e manifestos locais ficam ignorados.
As pastas `data/tactical/` e `data/broadcast/` são recriadas no clone porque seus guias são versionáveis.
Não há download automático nem upload de dados neste procedimento.
