"""Treina e valida, POR PARTIDA, o candidato de posse em jogos externos.

    python -m src.posse_ml.train --data outputs/skillcorner/m --out outputs/modelos/posse_ml.pkl

Para cada partida de teste: o modelo vê só as outras partidas. A comparação é pelo mesmo
caminho da frente 4 (candidato -> máquina de estados -> scorer oficial), contra a regra geométrica.
O rótulo é a posse do próprio tracking (SkillCorner), com 'bola solta mantém a posse'.
"""
import argparse
import glob
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

from src.frente4.fsm import Params, possession_labels  # noqa: E402
from src.frente4.pipeline import candidates  # noqa: E402
from src.frente4.features import compute_features, NOBODY  # noqa: E402
from src.posse_ml import features as F  # noqa: E402
from src.posse_ml.skillcorner import load_match, hold_labels  # noqa: E402

CONF = 0.6   # P(time) mínima para virar candidato; abaixo disso o candidato é "ninguém" (a máquina mantém)


def fsm_params(fps):
    """Parâmetros de frente4_v1.json (a 25 fps) reescalados para o fps do tracking."""
    s = fps / 25.0
    return Params(min_switch_frames=max(1, round(12 * s)), max_gap_frames=max(1, round(5 * s)),
                  restart_confirm_frames=max(1, round(3 * s)), min_dead_frames=max(1, round(5 * s)))


def new_model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_leaf_nodes=24,
                                          min_samples_leaf=200, l2_regularization=1.0, random_state=0)


def to_candidate(p_home, conf=CONF):
    cand = np.full(len(p_home), NOBODY, dtype=np.int8)
    cand[p_home >= conf] = 0
    cand[p_home <= 1 - conf] = 1
    return cand


def labels_to_scored(codes, truth, blocks):
    """GT e previsão em códigos 0/1/3 (X fora dos blocos) para pm.evaluate."""
    gt = np.full(len(truth), 3, dtype=np.int8)
    pred = np.full(len(truth), 3, dtype=np.int8)
    for a, b in blocks:
        gt[a:b + 1] = truth[a:b + 1]
        pred[a:b + 1] = codes[a:b + 1]
    return gt, pred


def score(codes, truth, blocks):
    import possession_metrics as pm
    gt, pred = labels_to_scored(codes, truth, blocks)
    r = pm.evaluate(gt, pred, bootstrap_n=0)
    return r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs", "modelos", "posse_ml.pkl"))
    ap.add_argument("--max-matches", type=int, default=0)
    args = ap.parse_args(argv)

    log = os.path.join(args.data, "..", "dl.log")
    finished = set(l.split()[1] for l in open(log) if l.startswith("done ")) if os.path.isfile(log) else set()
    matches = []
    for tj in sorted(glob.glob(os.path.join(args.data, "*_tracking.jsonl"))):
        mid = os.path.basename(tj).split("_")[0]
        mj = os.path.join(args.data, mid + "_match.json")
        if not (os.path.isfile(os.path.join(args.data, mid + ".ok")) or mid in finished):
            continue
        d = load_match(mj, tj)
        X, names = F.build(d)
        d["X"] = X
        d["truth"] = hold_labels(d["possession"], d["blocks"])
        matches.append(d)
        print("partida {}: {} frames, bola vista {:.0%}, posse rotulada {:.0%}".format(
            d["match_id"], len(X), float(np.mean(~np.isnan(d["ball_xy"][:, 0]))),
            float(np.mean(d["possession"] >= 0))), flush=True)
        if args.max_matches and len(matches) >= args.max_matches:
            break
    if len(matches) < 3:
        raise SystemExit("poucas partidas completas ({}); espere o download".format(len(matches)))

    rows = []
    for i, test in enumerate(matches):
        train = [m for j, m in enumerate(matches) if j != i]
        Xtr = np.vstack([m["X"][m["possession"] >= 0][::2] for m in train])
        ytr = np.concatenate([(m["possession"][m["possession"] >= 0][::2] == 0) for m in train]).astype(int)
        model = new_model().fit(Xtr, ytr)
        p_home = model.predict_proba(test["X"])[:, 1]
        params = fsm_params(test["fps"])
        dead = np.zeros(len(p_home), dtype=bool)
        cand_ml = to_candidate(p_home)
        feat = compute_features(test)
        cand_geo = candidates(feat, dict(contested_margin_m=0.5, max_control_speed_mps=10.0))
        lab = test["truth"]
        lab_mask = (test["possession"] >= 0)
        frame_acc = float(np.mean((p_home[lab_mask] >= 0.5) == (test["possession"][lab_mask] == 0)))
        geo_known = cand_geo[lab_mask] >= 0
        geo_acc = float(np.mean(cand_geo[lab_mask][geo_known] == test["possession"][lab_mask][geo_known]))
        res = {}
        for name, cand in (("geom", cand_geo), ("ml", cand_ml)):
            codes = possession_labels(cand, dead, test["blocks"], params)
            res[name] = score(codes, lab, test["blocks"])
        rows.append((test["match_id"], frame_acc, geo_acc, res["geom"], res["ml"]))
        print("teste {}: acc/frame ml {:.3f} (geom nos frames com candidato {:.3f}) | spellF1 geom {:.3f} ml {:.3f} | macroF1 geom {:.3f} ml {:.3f} | posses ml {} gt {}".format(
            test["match_id"], frame_acc, geo_acc, res["geom"][0], res["ml"][0], res["geom"][1], res["ml"][1],
            res["ml"][2], res["ml"][3]), flush=True)
    g = np.array([r[3][:2] for r in rows]); m = np.array([r[4][:2] for r in rows])
    print("\nMÉDIA em {} partidas (leave-one-match-out)".format(len(rows)))
    print("  spellF1  geométrico {:.4f}   ML {:.4f}".format(g[:, 0].mean(), m[:, 0].mean()))
    print("  macroF1  geométrico {:.4f}   ML {:.4f}".format(g[:, 1].mean(), m[:, 1].mean()))
    print("  ML melhor em spellF1 em {}/{} partidas".format(int((m[:, 0] > g[:, 0]).sum()), len(rows)))

    Xall = np.vstack([d["X"][d["possession"] >= 0][::2] for d in matches])
    yall = np.concatenate([(d["possession"][d["possession"] >= 0][::2] == 0) for d in matches]).astype(int)
    final = new_model().fit(Xall, yall)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as handle:
        pickle.dump(dict(model=final, names=names, conf=CONF, matches=[d["match_id"] for d in matches]), handle)
    print("modelo final salvo em", args.out)


if __name__ == "__main__":
    main()
