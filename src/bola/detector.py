"""Experimento: um detector pré-treinado (YOLO, classe COCO "sports ball") recupera a bola nos
frames em que o tracker a perdeu?

Duas medidas, no trecho público (frames 11185-18684):
  - CONTROLE: frames em que o tracker viu a bola. O detector acha a bola perto de onde o tracker
    a pôs (projetada no vídeo pela homografia)? Mede se o detector funciona nesta câmera.
  - BURACOS: frames em que o tracker NÃO viu a bola. Em quantos o detector acha algo dentro do
    campo? É um limite superior do que ele poderia recuperar.

Nenhuma anotação manual e nenhum gabarito: a referência do controle é o próprio tracker.

    python -m src.bola.detector                         # amostra pequena, CPU (~10-20 min)
    python -m src.bola.detector --passo 1 --modelo yolo11x.pt   # tudo, para GPU (Colab)

Saída: outputs/bola/detector_<modelo>.json com as detecções, e o resumo no terminal.
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from games import require  # noqa: E402
from src.bola.carregar import carregar, em_metros  # noqa: E402
from src.bola.limpar import limpar, trechos  # noqa: E402

# Params
JANELA = (11185, 18685)
CLASSE_BOLA = 32          # "sports ball" no COCO
CONF_MIN = 0.05
RAIOS_PX = (20, 40)


def homografias(caminho):
    hom = pd.read_csv(caminho)
    return hom[[c for c in hom.columns if c.startswith("h")]].to_numpy().reshape(-1, 3, 3)


def projetar(H, x, y):
    q = H @ np.array([x, y, 1.0])
    return q[:2] / q[2]


def no_campo(H, px, py, margem=0.02):
    """O pixel corresponde a um ponto dentro do campo (coordenadas normalizadas 0-1)?"""
    q = np.linalg.inv(H) @ np.array([px, py, 1.0])
    x, y = q[:2] / q[2]
    return -margem <= x <= 1 + margem and -margem <= y <= 1 + margem


def escolher_frames(fonte, passo):
    a0, a1 = JANELA
    idx = np.arange(a0, a1)
    controle = idx[(fonte[a0:a1] == "observado") & ((idx - a0) % (7 * passo) == 0)]
    buracos = []
    for a, z in zip(*trechos(fonte[a0:a1] != "observado")):
        if z - a >= 13:
            buracos.append((a + a0, z + a0))
    frames_buraco = np.concatenate([np.arange(a, z, passo) for a, z in buracos])
    return controle, frames_buraco, buracos


def detectar(video, frames, modelo, imgsz):
    from ultralytics import YOLO
    yolo = YOLO(modelo)
    cap = cv2.VideoCapture(video)
    out, t0 = {}, time.time()
    for k, f in enumerate(sorted(int(f) for f in frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            raise RuntimeError("falha lendo o frame {}".format(f))
        r = yolo.predict(img, classes=[CLASSE_BOLA], conf=CONF_MIN, imgsz=imgsz, verbose=False)[0]
        out[f] = [(float(b[0]), float(b[1]), float(c))
                  for b, c in zip(r.boxes.xywh.cpu().numpy(), r.boxes.conf.cpu().numpy())]
        if k % 50 == 0:
            print("  {}/{} frames, {:.1f} s/frame".format(k, len(frames), (time.time() - t0) / (k + 1)),
                  flush=True)
    cap.release()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="yolo11m.pt")
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--passo", type=int, default=6, help="usa 1 a cada N frames (1 = todos)")
    args = ap.parse_args()

    video, hom_csv = require(["video", "homography"])
    dados = carregar()
    por_frame, _, _ = limpar(dados, estimar_longos=False)
    fonte = por_frame["fonte"].to_numpy()
    bola = dados["bola"][:, :2]
    Hs = homografias(hom_csv)
    controle, frames_buraco, buracos = escolher_frames(fonte, args.passo)
    print("controle: {} frames | buracos: {} frames de {} buracos".format(
        len(controle), len(frames_buraco), len(buracos)))

    det = detectar(video, np.r_[controle, frames_buraco], args.modelo, args.imgsz)

    dist = []
    for f in controle:
        e = projetar(Hs[f], *bola[f])
        ds = [np.hypot(x - e[0], y - e[1]) for x, y, _ in det[f]]
        dist.append(min(ds) if ds else np.inf)
    dist = np.array(dist)
    print("\nCONTROLE (tracker viu a bola), n={}".format(len(dist)))
    print("  detectou algo: {:.1%}".format(np.isfinite(dist).mean()))
    for r in RAIOS_PX:
        print("  detecção a <= {} px da bola do tracker: {:.1%}".format(r, (dist <= r).mean()))

    print("\nBURACOS (tracker não viu a bola)")
    total = com = 0
    for a, z in buracos:
        fr = [f for f in frames_buraco if a <= f < z]
        n = sum(1 for f in fr if any(no_campo(Hs[f], x, y) for x, y, _ in det[f]))
        total += len(fr)
        com += n
        print("  {}-{} ({:.1f} s): detecção dentro do campo em {:.0%} dos frames".format(
            a, z, (z - a) / 25, n / max(len(fr), 1)))
    print("  TOTAL: {:.1%} de {} frames".format(com / max(total, 1), total))

    saida = os.path.join(ROOT, "outputs", "bola", "detector_{}.json".format(
        os.path.splitext(os.path.basename(args.modelo))[0]))
    os.makedirs(os.path.dirname(saida), exist_ok=True)
    json.dump({"modelo": args.modelo, "imgsz": args.imgsz, "passo": args.passo,
               "deteccoes": {str(k): v for k, v in det.items()}}, open(saida, "w"))
    print("\nsalvo em", saida)


if __name__ == "__main__":
    main()
