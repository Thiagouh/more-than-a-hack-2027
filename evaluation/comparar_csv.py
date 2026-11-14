"""Compare previsões sem consultar gabarito nem atribuir nota de qualidade."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import possession_metrics as pm
from validate_submission import describe


def load(path, rows=None):
    table = pm.read_label_csv(path)
    if table.empty:
        raise ValueError('CSV vazio')
    reason = pm.validate_submission(table, rows if rows is not None else len(table))
    if reason:
        raise ValueError('{}: {}'.format(path, reason))
    return pm.label_codes(table)


def summarize(codes, fps):
    count, duration = describe(codes, fps)
    return {
        'frames': len(codes),
        'labels': {name: int(np.sum(codes == i)) for i, name in enumerate(pm.LABELS)},
        'posses_brutas': count,
        'duracao_media_posse_bruta_s': duration,
        'aviso': 'X interrompe blocos neste resumo bruto; não é a contagem oficial com gabarito.',
    }


def compare(left, right, fps=25):
    if len(left) != len(right):
        raise ValueError('Os arquivos não cobrem os mesmos frames')
    different = left != right
    edges = np.diff(np.r_[0, different.astype(int), 0])
    intervals = []
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        intervals.append({'inicio': int(start), 'fim_inclusivo': int(stop - 1),
                          'frames': int(stop - start), 'duracao_s': (stop - start) / fps})
    matrix = {a: {b: int(np.sum((left == i) & (right == j)))
                  for j, b in enumerate(pm.LABELS)} for i, a in enumerate(pm.LABELS)}
    return {'a': summarize(left, fps), 'b': summarize(right, fps),
            'frames_diferentes': int(different.sum()),
            'percentual_diferente': float(100 * different.mean()),
            'cruzamento_a_linhas_b_colunas': matrix, 'intervalos': intervals}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('a', type=Path)
    parser.add_argument('b', type=Path)
    parser.add_argument('--rows', type=int, help='Total esperado; sem isso, valida apenas cobertura mútua')
    parser.add_argument('--fps', type=float, default=25)
    parser.add_argument('--out', type=Path, required=True, help='Diretório para resumo e intervalos')
    args = parser.parse_args()
    if not np.isfinite(args.fps) or args.fps <= 0:
        parser.error('--fps deve ser positivo e finito')
    if args.rows is not None and args.rows <= 0:
        parser.error('--rows deve ser positivo')
    try:
        report = compare(load(args.a, args.rows), load(args.b, args.rows), args.fps)
    except (ValueError, OSError) as error:
        parser.exit(1, 'Erro: {}\n'.format(error))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'comparacao.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    with (args.out / 'divergencias.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['inicio', 'fim_inclusivo', 'frames', 'duracao_s'])
        writer.writeheader()
        writer.writerows(report['intervalos'])
    print('{} frames diferentes ({:.2f}%), em {} intervalos.'.format(
        report['frames_diferentes'], report['percentual_diferente'], len(report['intervalos'])))
    print('Diferença não indica melhora. Resumo bruto: X interrompe blocos; sem nota oficial.')
    print('Relatórios em {}'.format(args.out))


if __name__ == '__main__':
    main()
