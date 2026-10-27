"""Conferir arquivos locais esperados, sem baixar ou modificar dados da partida."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from games import asset_paths  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", choices=("tactical", "broadcast", "all"), default="tactical",
        help="tactical por padrão; all inclui a fonte broadcast opcional",
    )
    parser.add_argument(
        "--data-dir", type=Path,
        help="pasta que contém tactical/ e broadcast/; o padrão usa MTAH_DATA_DIR ou data/",
    )
    args = parser.parse_args()
    sources = ("tactical", "broadcast") if args.source == "all" else (args.source,)
    missing = 0
    total = 0
    for source in sources:
        print("\n{}".format(source))
        for path in asset_paths(data_dir=args.data_dir, source=source).values():
            path = Path(path)
            total += 1
            if path.is_file() and path.stat().st_size > 0:
                print("  OK      {}".format(path))
            else:
                missing += 1
                print("  AUSENTE OU VAZIO  {}".format(path))
    if missing:
        print("\nFaltam {} de {} arquivos. Consulte docs/DADOS-LOCAIS.md.".format(missing, total))
        return 1
    print("\n{} arquivos presentes. A conferência não valida o conteúdo ou o alinhamento.".format(total))
    return 0


if __name__ == "__main__":
    sys.exit(main())
