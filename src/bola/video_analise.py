"""Vídeo de análise: gabarito público + tracking projetado no vídeo + trilha limpa da bola.

Por padrão renderiza a janela pública inteira (frames 11185-18684, 5 min). Desenha sobre cada frame:
  - jogadores do ATD projetados pela homografia (H azul, A vermelho);
  - bola: observada (amarelo), descartada como falha (X magenta, posição bruta), estimada
    (círculo laranja vazado: é uma região, não um ponto);
  - painel com frame, tempo, rótulo do gabarito, fonte da bola e distância do jogador mais
    próximo de cada time até a bola;
  - minimapa e duas faixas de linha do tempo (gabarito e fonte da bola) com cursor.

Tudo fica em outputs/ (ignorado pelo Git). Nada é enviado a lugar nenhum.

    python -m src.bola.video_analise                         # 5 min, 1280 px
    python -m src.bola.video_analise --inicio 12800 --segundos 30
"""

import argparse
import os
import subprocess
import warnings
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require  # noqa: E402
from src.bola.carregar import carregar, em_metros  # noqa: E402
from src.bola.limpar import limpar  # noqa: E402

# Cores BGR
COR_TIME = {"H": (255, 150, 50), "A": (80, 80, 255)}
COR_ROTULO = {"H": (255, 150, 50), "A": (80, 80, 255), "D": (150, 150, 150), "X": (40, 40, 40)}
COR_FONTE = {"observado": (0, 230, 255), "preenchido": (0, 255, 0), "estimado": (0, 140, 255),
             "ausente": (60, 60, 60), "falha": (255, 0, 255)}
NOME_ROTULO = {"H": "H  Barcelona", "A": "A  Athletic", "D": "D  bola morta", "X": "X  fora da janela"}
FONTE = cv2.FONT_HERSHEY_SIMPLEX


def janela_publica(gt):
    idx = np.flatnonzero(gt != "X")
    return int(idx[0]), int(idx[-1]) + 1


def projetar(H, xy_norm, escala):
    x, y = xy_norm[..., 0], xy_norm[..., 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        den = H[2, 0] * x + H[2, 1] * y + H[2, 2]
        px = (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / den
        py = (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / den
    return np.stack([px * escala[0], py * escala[1]], -1)


def ponto(p):
    return tuple(int(v) for v in np.clip(np.rint(p), -10000, 10000))


def faixa(rotulos, cores, largura, altura):
    """Imagem (altura, largura) colorindo uma sequência de rótulos comprimida na largura."""
    n = len(rotulos)
    col = (np.arange(largura) * n / largura).astype(int)
    img = np.zeros((altura, largura, 3), np.uint8)
    for k, c in cores.items():
        img[:, rotulos[col] == k] = c
    return img


def texto(img, s, org, escala=0.55, cor=(255, 255, 255), esp=1):
    cv2.putText(img, s, org, FONTE, escala, (0, 0, 0), esp + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, FONTE, escala, cor, esp, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inicio", type=int, default=None, help="frame inicial (padrão: janela pública)")
    ap.add_argument("--segundos", type=float, default=None, help="duração (padrão: até o fim da janela)")
    ap.add_argument("--largura", type=int, default=1280)
    ap.add_argument("--saida", default=os.path.join(ROOT, "outputs", "videos", "analise_janela_publica.mp4"))
    args = ap.parse_args()

    video, hom_csv = require(["video", "homography"])
    gt = pd.read_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))["label"].to_numpy()
    w0, w1 = janela_publica(gt)
    inicio = w0 if args.inicio is None else args.inicio
    fim = w1 if args.segundos is None else min(len(gt), inicio + int(round(args.segundos * 25)))

    print("carregando tracking e limpando a bola...")
    dados = carregar()
    limpa, _, falha = limpar(dados)
    times = dados["times"]
    jog_n = dados["jogadores"][..., :2]
    bola_bruta_n = dados["bola"][:, :2]
    pitch = np.asarray(GAMES[GAME]["pitch"])
    bola_limpa_n = limpa[["x", "y"]].to_numpy() / pitch
    fonte = limpa["fonte"].to_numpy().astype(object)
    fonte[falha] = "falha"
    fonte = fonte.astype(str)

    # distâncias (m) do jogador mais próximo de cada time à bola limpa
    jog_m = em_metros(jog_n).astype(np.float64)
    bola_m = limpa[["x", "y"]].to_numpy()
    d = np.linalg.norm(jog_m - bola_m[:, None, :], axis=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dist_time = {t: np.nanmin(np.where(times[None, :] == t, d, np.nan), axis=1) for t in "HA"}

    hom = pd.read_csv(hom_csv)
    Hs = hom[[c for c in hom.columns if c.startswith("h")]].to_numpy().reshape(-1, 3, 3)

    cap = cv2.VideoCapture(video)
    sw, sh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = args.largura // 2 * 2
    Hh = int(round(sh * W / sw / 2) * 2)
    escala = np.array([W / sw, Hh / sh])
    cap.set(cv2.CAP_PROP_POS_FRAMES, inicio)
    if abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - inicio) > 0.5:
        raise RuntimeError("não foi possível posicionar o vídeo no frame {}".format(inicio))

    # faixas da linha do tempo (pré-renderizadas)
    FX = 14
    faixa_gt = faixa(gt[inicio:fim], COR_ROTULO, W, FX)
    faixa_fonte = faixa(fonte[inicio:fim], COR_FONTE, W, FX)
    altura_total = Hh + 2 * FX + 24

    # minimapa
    mw = W // 4
    mh = int(mw * pitch[1] / pitch[0])
    pad = 6

    def mm(p):
        return ponto(pad + np.asarray(p) * [mw - 2 * pad, mh - 2 * pad])

    os.makedirs(os.path.dirname(args.saida), exist_ok=True)
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", "{}x{}".format(W, altura_total), "-r", "25", "-i", "pipe:0", "-an", "-c:v", "libx264",
         "-preset", "veryfast", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         args.saida], stdin=subprocess.PIPE)

    total = fim - inicio
    try:
        for i, f in enumerate(range(inicio, fim)):
            ok, img = cap.read()
            if not ok:
                raise RuntimeError("falha lendo o vídeo no frame {}".format(f))
            img = cv2.resize(img, (W, Hh), interpolation=cv2.INTER_AREA)
            H = Hs[f]

            # jogadores
            pj = projetar(H, jog_n[f], escala)
            for s in range(len(times)):
                if np.isfinite(pj[s]).all():
                    cv2.circle(img, ponto(pj[s]), 7, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.circle(img, ponto(pj[s]), 7, COR_TIME[times[s]], 2, cv2.LINE_AA)

            # bola
            fo = fonte[f]
            if fo == "falha" and np.isfinite(bola_bruta_n[f]).all():
                c = ponto(projetar(H, bola_bruta_n[f], escala))
                cv2.drawMarker(img, c, COR_FONTE["falha"], cv2.MARKER_TILTED_CROSS, 18, 2)
            if np.isfinite(bola_limpa_n[f]).all() and fo != "falha":
                c = ponto(projetar(H, bola_limpa_n[f], escala))
                if fo == "estimado":
                    cv2.circle(img, c, 22, COR_FONTE["estimado"], 2, cv2.LINE_AA)
                    texto(img, "est.", (c[0] + 24, c[1]), 0.45, COR_FONTE["estimado"])
                else:
                    cv2.circle(img, c, 6, (0, 0, 0), -1, cv2.LINE_AA)
                    cv2.circle(img, c, 4, COR_FONTE[fo], -1, cv2.LINE_AA)

            # painel
            rot = gt[f]
            cv2.rectangle(img, (0, 0), (470, 122), (20, 20, 20), -1)
            t = f / 25
            texto(img, "frame {}   video {:02d}:{:05.2f}".format(f, int(t // 60), t % 60), (16, 30))
            cv2.rectangle(img, (16, 40), (36, 60), COR_ROTULO[rot], -1)
            texto(img, "gabarito: " + NOME_ROTULO[rot], (44, 57), 0.6, (255, 255, 255), 2)
            texto(img, "bola: " + fo, (16, 84), 0.55, COR_FONTE[fo])
            dh, da = dist_time["H"][f], dist_time["A"][f]
            texto(img, "mais perto da bola:  H {}   A {}".format(
                "-" if np.isnan(dh) else "{:.1f} m".format(dh),
                "-" if np.isnan(da) else "{:.1f} m".format(da)), (16, 108), 0.5)

            # minimapa
            painel = np.full((mh, mw, 3), (35, 70, 35), np.uint8)
            cv2.rectangle(painel, mm((0, 0)), mm((1, 1)), (170, 200, 170), 1)
            cv2.line(painel, mm((0.5, 0)), mm((0.5, 1)), (170, 200, 170), 1)
            for s in range(len(times)):
                if np.isfinite(jog_n[f, s]).all():
                    cv2.circle(painel, mm(jog_n[f, s]), 3, COR_TIME[times[s]], -1, cv2.LINE_AA)
            if fo == "falha" and np.isfinite(bola_bruta_n[f]).all():
                cv2.drawMarker(painel, mm(bola_bruta_n[f]), COR_FONTE["falha"], cv2.MARKER_TILTED_CROSS, 8, 1)
            elif np.isfinite(bola_limpa_n[f]).all():
                if fo == "estimado":
                    raio = int(10 / pitch[0] * (mw - 2 * pad))  # ~10 m, o erro típico
                    cv2.circle(painel, mm(bola_limpa_n[f]), raio, COR_FONTE["estimado"], 1, cv2.LINE_AA)
                else:
                    cv2.circle(painel, mm(bola_limpa_n[f]), 3, COR_FONTE[fo], -1, cv2.LINE_AA)
            y0, x0 = Hh - mh - 8, W - mw - 8
            img[y0:y0 + mh, x0:x0 + mw] = cv2.addWeighted(painel, 0.85, img[y0:y0 + mh, x0:x0 + mw], 0.15, 0)

            # linhas do tempo
            barra = np.zeros((2 * FX + 24, W, 3), np.uint8)
            barra[2:2 + FX] = faixa_gt
            barra[4 + FX:4 + 2 * FX] = faixa_fonte
            cx = int(i * W / total)
            cv2.line(barra, (cx, 0), (cx, 2 * FX + 6), (255, 255, 255), 2)
            texto(barra, "cima: gabarito (azul H, vermelho A, cinza D)   baixo: bola (amarelo obs., laranja est., "
                         "magenta falha)", (6, 2 * FX + 20), 0.4, (200, 200, 200))
            enc.stdin.write(np.vstack([img, barra]).tobytes())
            if i % 750 == 0:
                print("  {}/{} frames".format(i, total), flush=True)
    finally:
        cap.release()
        enc.stdin.close()
        enc.wait()
    print("salvo em", args.saida)


if __name__ == "__main__":
    main()
