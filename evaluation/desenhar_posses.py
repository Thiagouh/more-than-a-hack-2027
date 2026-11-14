"""Desenhe CSVs de posse no mesmo eixo, preservando labels e tempo originais."""
import argparse
from html import escape
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
from comparar_csv import load
from possession_metrics import LABELS

COLORS = {'H': '#2563eb', 'A': '#e85d32', 'D': '#64748b', 'X': '#e5b82e'}


def intervals(codes, start, end):
    """Retorne (início, fim exclusivo, label), recortados a limites inclusivos."""
    window = codes[start:end + 1]
    edges = np.r_[0, np.flatnonzero(window[1:] != window[:-1]) + 1, len(window)]
    return [(start + int(a), start + int(b), LABELS[int(window[a])])
            for a, b in zip(edges[:-1], edges[1:])]


def render_svg(series, start=0, end=None, fps=25):
    if not series:
        raise ValueError('Informe pelo menos um CSV')
    size = len(series[0][1])
    if not size or any(len(codes) != size for _, codes in series):
        raise ValueError('Os arquivos devem ter cobertura igual e não vazia')
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('FPS deve ser positivo e finito')
    end = size - 1 if end is None else end
    if not 0 <= start <= end < size:
        raise ValueError('Intervalo inválido: use 0 <= inicio <= fim < {}'.format(size))
    width, left, right, top, row = 1280, 240, 48, 130, 78
    height = top + row * len(series) + 115
    span = width - left - right
    def x(frame):
        return left + (frame - start) / (end + 1 - start) * span
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="#f8fafc"/>',
             '<g font-family="Arial, sans-serif" fill="#172033">',
             '<text x="32" y="38" font-size="24" font-weight="bold">Linha do tempo de posse</text>',
             f'<text x="32" y="65" font-size="14">Frames {start}–{end} (inclusivos) · {fps:g} fps · labels originais, sem preencher X</text>']
    for i, label in enumerate(LABELS):
        px = 32 + i * 280
        caption = {'H': 'H · Barcelona', 'A': 'A · Athletic Club', 'D': 'D · Bola morta', 'X': 'X · Indeterminado'}[label]
        parts += [f'<rect x="{px}" y="84" width="18" height="18" rx="3" fill="{COLORS[label]}"/>',
                  f'<text x="{px + 27}" y="98" font-size="14">{caption}</text>']
    bottom = top + row * len(series) - 20
    for tick in range(7):
        frame = start + tick / 6 * (end + 1 - start)
        px = x(frame)
        parts += [f'<path d="M{px:.3f},{top - 10} V{bottom}" stroke="#cbd5e1"/>',
                  f'<text x="{px:.3f}" y="{bottom + 28}" text-anchor="middle" font-size="12">{frame / fps:.2f}</text>']
    for i, (name, codes) in enumerate(series):
        y = top + row * i
        short = name if len(name) <= 26 else name[:23] + '…'
        parts.append(f'<text x="{left - 15}" y="{y + 26}" text-anchor="end" font-size="14"><title>{escape(name)}</title>{escape(short)}</text>')
        for a, b, label in intervals(codes, start, end):
            parts.append(f'<rect x="{x(a):.3f}" y="{y}" width="{x(b)-x(a):.3f}" height="42" fill="{COLORS[label]}"><title>{label}: frames {a}–{b-1}</title></rect>')
    parts += [f'<text x="{left + span / 2}" y="{bottom + 58}" text-anchor="middle" font-size="14">Tempo desde o frame 0 (segundos)</text>',
              f'<text x="32" y="{height - 20}" font-size="12" fill="#526174">Visualização de previsões: diferenças entre faixas não indicam qual versão está correta.</text>', '</g></svg>']
    return '\n'.join(parts)


def save(svg, out):
    if out.suffix.lower() not in ('.svg', '.png'):
        raise ValueError('--out deve terminar em .svg ou .png')
    converter = shutil.which('rsvg-convert')
    if out.suffix.lower() == '.png' and not converter:
        raise ValueError('PNG requer rsvg-convert (librsvg). Use --out arquivo.svg como alternativa.')
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == '.svg':
        out.write_text(svg, encoding='utf-8')
    else:
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'timeline.svg'
            source.write_text(svg, encoding='utf-8')
            subprocess.run([converter, str(source), '-o', str(out)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csvs', type=Path, nargs='+')
    parser.add_argument('--inicio', type=int, default=0)
    parser.add_argument('--fim', type=int)
    parser.add_argument('--fps', type=float, default=25)
    parser.add_argument('--rows', type=int)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.rows is not None and args.rows <= 0:
            raise ValueError('--rows deve ser positivo')
        series = [(path.stem, load(path, args.rows)) for path in args.csvs]
        save(render_svg(series, args.inicio, args.fim, args.fps), args.out)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'Erro: {error}\n')
    print(f'Gráfico salvo em {args.out}')


if __name__ == '__main__':
    main()
