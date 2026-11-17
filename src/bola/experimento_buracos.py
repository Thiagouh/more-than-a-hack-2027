"""Quanto da nota se perde nos buracos da bola? Compara, com a métrica oficial no trecho público,
o baseline da organização com variantes que só mudam o tratamento dos buracos.

Atenção: o trecho público tem 17 posses. Serve para ver a ordem de grandeza do efeito, não para
escolher parâmetros (ver docs/EVALUATION.md).

    python -m src.bola.experimento_buracos
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from src.bola.carregar import carregar, em_metros  # noqa: E402
from src.bola.limpar import limpar  # noqa: E402
from starter.baseline import smooth  # noqa: E402

HOME, AWAY, NOBODY = 0, 1, -1
JANELA = 50  # a do baseline publicado (smooth50)


def mais_proximo(bola_m, jog_m, times):
    """Código do time do jogador mais próximo da bola; NOBODY onde não há bola."""
    d = np.linalg.norm(jog_m - bola_m[:, None, :], axis=2)
    d = np.where(np.isfinite(d), d, np.inf)
    idx = d.argmin(1)
    codigo = np.where(times[idx] == "H", HOME, AWAY).astype(np.int8)
    codigo[~np.isfinite(bola_m[:, 0])] = NOBODY
    return codigo


def herdar(codigo, mascara):
    """Nos frames de `mascara` sem resposta, repete o último código conhecido."""
    out = codigo.copy()
    idx = np.where(out != NOBODY, np.arange(len(out)), 0)
    np.maximum.accumulate(idx, out=idx)
    preencher = mascara & (out == NOBODY)
    out[preencher] = out[idx[preencher]]
    return out


def avaliar(nome, rotulos, gt_codes):
    r = pm.evaluate(gt_codes, pm.label_codes(rotulos), bootstrap_n=0)
    return {"variante": nome, "spellF1": r["spell_f1_mean"], "macroF1": r["macro_f1"],
            "posses_previstas": r["n_pred_spells"], "posses_gt": r["n_gt_spells"],
            "acc": r["accuracy"]}


def main():
    dados = carregar()
    times = dados["times"]
    jog_m = em_metros(dados["jogadores"][..., :2]).astype(np.float64)
    bola_bruta = em_metros(dados["bola"][:, :2]).astype(np.float64)
    limpa, buracos, _ = limpar(dados)
    bola_limpa = limpa[["x", "y"]].to_numpy()
    fonte = limpa["fonte"].to_numpy()
    gt = pd.read_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))
    gt_codes = pm.label_codes(gt)

    # buracos em que a bola sumiu fora do campo -> D
    saiu = np.zeros(len(bola_limpa), bool)
    for _, b in buracos[buracos["sumiu_fora_do_campo"]].iterrows():
        saiu[int(b.inicio):int(b.fim)] = True

    tudo = np.ones(len(bola_limpa), bool)
    bruto = mais_proximo(bola_bruta, jog_m, times)
    so_obs = np.where(np.isin(fonte, ["observado", "preenchido"]), 1, 0).astype(bool)
    limpo = mais_proximo(np.where(so_obs[:, None], bola_limpa, np.nan), jog_m, times)
    estimado = mais_proximo(bola_limpa, jog_m, times)

    from src.bola.modelo import prever_buracos
    xy_modelo, prob_h = prever_buracos()
    buraco = np.isfinite(xy_modelo[:, 0])
    com_modelo = mais_proximo(np.where(buraco[:, None], xy_modelo, bola_limpa), jog_m, times)
    com_modelo[~buraco & ~so_obs] = NOBODY
    herdado = herdar(limpo, tudo)

    def herda_mas_troca(conf):
        out = herdado.copy()
        alt = buraco & (np.maximum(prob_h, 1 - prob_h) >= conf)
        out[alt] = np.where(prob_h[alt] > 0.5, HOME, AWAY)
        return np.where(saiu, NOBODY, out)

    from src.bola.troca import maioria, rotular, JANELA_PONTA

    def por_buraco(estrategia):
        """Cada buraco real recebe antes/depois e a estratégia de troca; saiu do campo = D."""
        out = limpo.copy()
        cod01 = np.where(limpo == HOME, 1, np.where(limpo == AWAY, 0, -1))
        for _, b in buracos.iterrows():
            a, z = int(b.inicio), int(b.fim)
            if b.frames <= 12 or a == 0 or z >= len(out):
                continue
            if b.sumiu_fora_do_campo:
                out[a:z] = NOBODY
                continue
            antes = maioria(cod01[max(0, a - JANELA_PONTA):a])
            depois = maioria(cod01[z:z + JANELA_PONTA])
            p = prob_h[a:z]
            if np.isnan(p).any():
                p = np.full(z - a, 0.5)
            r = rotular(antes, depois, z - a, estrategia, p)
            out[a:z] = np.where(r == 1, HOME, np.where(r == 0, AWAY, NOBODY))
        return out

    variantes = {
        "1 baseline (buraco = D)": bruto,
        "2 bola limpa (buraco = D)": limpo,
        "3 limpa + buraco herda posse": herdar(limpo, tudo),
        "4 limpa + herda, mas saiu do campo = D": np.where(saiu, NOBODY, herdar(limpo, tudo)),
        "5 limpa + bola estimada nos buracos": estimado,
        "6 estimada, mas saiu do campo = D": np.where(saiu, NOBODY, estimado),
        "7 posição do modelo A, saiu = D": np.where(saiu, NOBODY, com_modelo),
        "8 herda; troca se modelo A conf>=0.8": herda_mas_troca(0.8),
        "9 herda; troca se modelo A conf>=0.9": herda_mas_troca(0.9),
        "10 antes/depois, troca no fim": por_buraco("fim"),
        "11 antes/depois, troca no meio": por_buraco("meio"),
        "12 antes/depois, troca pelo modelo": por_buraco("modelo"),
    }
    linhas = [avaliar(n, smooth(c, JANELA), gt_codes) for n, c in variantes.items()]
    df = pd.DataFrame(linhas)
    pd.set_option("display.width", 140)
    print("trecho público, suavização ±{} frames (igual ao baseline smooth50)\n".format(JANELA))
    print(df.to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
