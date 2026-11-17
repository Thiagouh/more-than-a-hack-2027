"""CLI da frente 4.

    python -m src.frente4 run --out outputs/submissions/frente4.csv [--data-dir data] [--config configs/frente4_v1.json] [--events outputs/frente5/run/events.jsonl]
    python -m src.frente4 baseline --out outputs/submissions/baselines
    python -m src.frente4 from-csv evaluation/results/baseline_raw.csv --out outputs/submissions/dev.csv
    python -m src.frente4 score outputs/submissions/frente4.csv [outro.csv ...]

``score`` usa o scorer oficial contra a janela pública: é conferência de integração, não
critério para escolher parâmetros (docs/EVALUATION.md).
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from . import pipeline  # noqa: E402


def cmd_run(args):
    path, diag = pipeline.run(args.out, args.data_dir, args.config, args.events, args.cache_dir,
                              bola_path=args.bola, deadball_path=args.deadball)
    s = diag["summary"]
    print("escrito {}".format(path))
    print("times: H={} A={}".format(diag["teams"]["H"], diag["teams"]["A"]))
    print("posses {}  duração média {:.1f} s  labels {}".format(
        s["n_spells"], s["mean_duration_s"], s["counts"]))
    print("bola observada {:.1%}  candidato {:.1%}  D por motivo {}".format(
        diag["ball_observed_fraction"], diag["candidate_fraction"], diag["dead_reasons"]))


def cmd_baseline(args):
    from .atd import load_tracking
    from .baseline import reproduce
    tr = load_tracking(args.data_dir, args.cache_dir)
    ref = os.path.join(ROOT, "evaluation", "results")
    for name, (path, identical) in reproduce(tr, args.out, ref).items():
        print("{:<20} {}  idêntico ao oficial: {}".format(name, path, identical))


def cmd_from_csv(args):
    path, diag = pipeline.from_csv(args.csv, args.out, args.config)
    s = diag["summary"]
    print("escrito {}".format(path))
    print("posses {}  duração média {:.1f} s  labels {}".format(
        s["n_spells"], s["mean_duration_s"], s["counts"]))


def cmd_score(args):
    sys.path.insert(0, os.path.join(ROOT, "evaluation"))
    import possession_metrics as pm
    gt_path = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
    gt = pm.label_codes(pm.read_label_csv(gt_path))
    print("{:<40} {:>8} {:>8} {:>5} {:>6}".format("arquivo", "spellF1", "macroF1", "poss", "dur_s"))
    for path in args.csv:
        table = pm.read_label_csv(path)
        reason = pm.validate_submission(table, len(gt))
        if reason:
            print("{:<40} INVÁLIDO: {}".format(os.path.basename(path), reason))
            continue
        r = pm.evaluate(gt, pm.label_codes(table))
        print("{:<40} {:>8.4f} {:>8.4f} {:>5} {:>6.1f}".format(
            os.path.basename(path)[:40], r["spell_f1_mean"], r["macro_f1"],
            r["n_pred_spells"], r["mean_spell_duration_s"]))
    print("(gabarito público: 17 posses, 14,2 s; janela de 5 min, não serve para ajustar parâmetros)")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m src.frente4", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="gera a submissão a partir dos dados")
    p.add_argument("--out", default=os.path.join(ROOT, "outputs", "submissions", "frente4.csv"))
    p.add_argument("--data-dir")
    p.add_argument("--config")
    p.add_argument("--events", help="events.jsonl v2 da frente 5 (opcional)")
    p.add_argument("--bola", help="sinal_bola.csv da frente 2 (opcional); 'auto' usa outputs/bola/sinal_bola.csv")
    p.add_argument("--cache-dir")
    p.add_argument("--deadball", help="dead_ball_features.csv do detector Smart Tagging + bola morta (opcional)")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("baseline", help="refaz os quatro baselines e compara com evaluation/results")
    p.add_argument("--out", default=os.path.join(ROOT, "outputs", "submissions", "baselines"))
    p.add_argument("--data-dir")
    p.add_argument("--cache-dir")
    p.set_defaults(func=cmd_baseline)

    p = sub.add_parser("from-csv", help="máquina de estados sobre um CSV de candidatos")
    p.add_argument("csv")
    p.add_argument("--out", required=True)
    p.add_argument("--config")
    p.set_defaults(func=cmd_from_csv)

    p = sub.add_parser("score", help="nota na janela pública (conferência, não ajuste)")
    p.add_argument("csv", nargs="+")
    p.set_defaults(func=cmd_score)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
