"""Gere duas previsões artificiais; não representam uma partida real."""
import csv
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'demo_timeline'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = list('H' * 200 + 'A' * 175 + 'D' * 75 + 'H' * 150)
    noisy = base.copy()
    noisy[60:70] = 'A' * 10  # fragmentação
    noisy[120:140] = 'X' * 20  # ausência de evidência
    noisy[200:225] = 'H' * 25  # mudança atrasada em um segundo
    noisy[490:500] = 'D' * 10  # interrupção adicional
    paths = []
    for name, labels in [('cenario_coerente', base), ('cenario_com_falhas', noisy)]:
        path = OUT / f'{name}.csv'
        with path.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['frame', 'label'])
            writer.writerows(enumerate(labels))
        paths.append(str(path))
    for extension in ('svg', 'png'):
        subprocess.run([sys.executable, str(ROOT / 'evaluation/desenhar_posses.py'),
                        *paths, '--out', str(OUT / f'linha_do_tempo.{extension}')], check=True)


if __name__ == '__main__':
    main()
