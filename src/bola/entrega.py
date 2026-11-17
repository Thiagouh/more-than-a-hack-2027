"""Entrega da frente 2: um sinal da bola por frame, pronto para a máquina de estados (frente 4).

    python -m src.bola.entrega        # gera outputs/bola/sinal_bola.csv (~1 min) e confere a nota

Em código:

    from src.bola.entrega import carregar_sinal
    sinal = carregar_sinal()          # DataFrame com 199319 linhas, uma por frame do vídeo

Colunas
-------
frame            frame do vídeo tático (0..199318) = linha da submissão
x, y             posição da bola em metros (campo 105,3 x 68). NaN só fora dos tempos de jogo
fonte            observado | preenchido (buraco <= 0,5 s, reta) | estimado (buraco longo, modelo A)
                 | ausente
buraco_id        -1 fora de buraco; senão o id do buraco (ver outputs/bola/buracos.csv)
time_proximo     H/A: time do jogador mais próximo da bola (só onde a bola é observada/preenchida)
dist_h, dist_a   distância (m) do jogador mais próximo de cada time até a bola (observada ou estimada)
time_antes       H/A: maioria do time mais próximo nos 25 frames observados antes do buraco
time_depois      H/A: idem, nos 25 frames depois do buraco
saiu_campo       True se a última posição antes do buraco estava fora das linhas (pista de D)
prob_h           P(time mais próximo da bola = H) pelo modelo A, só nos buracos
posse_buraco     sugestão segura para o buraco: D se saiu_campo, senão time_antes
                 (validada no trecho público: spellF1 0,3256 -> 0,4205)
posse_buraco_modelo  alternativa: troca no ponto indicado pelo modelo A quando antes != depois
                 (melhor nos buracos sintéticos, pior no trecho público; ver README)

Nada aqui usa o gabarito. O gabarito público só é lido no __main__, para conferência.
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.bola.carregar import carregar, em_metros  # noqa: E402
from src.bola.limpar import limpar  # noqa: E402
from src.bola.troca import JANELA_PONTA, maioria, rotular  # noqa: E402

SAIDA = os.path.join(ROOT, "outputs", "bola", "sinal_bola.csv")
MIN_BURACO = 13  # frames; buracos menores já foram preenchidos por reta


def _dist_por_time(bola_m, jog_m, eh_h):
    d = np.linalg.norm(jog_m - bola_m[:, None, :], axis=2)
    d = np.where(np.isfinite(d), d, np.inf)
    dh = d[:, eh_h].min(1)
    da = d[:, ~eh_h].min(1)
    sem = ~np.isfinite(bola_m[:, 0])
    dh[sem | ~np.isfinite(dh)] = np.nan
    da[sem | ~np.isfinite(da)] = np.nan
    return dh, da


def gerar():
    from src.bola.modelo import prever_buracos

    dados = carregar()
    eh_h = dados["times"] == "H"
    jog_m = em_metros(dados["jogadores"][..., :2]).astype(np.float64)
    por_frame, buracos, _ = limpar(dados)  # posição pela regra antiga nos buracos (fallback)
    n = len(por_frame)

    xy = por_frame[["x", "y"]].to_numpy().copy()
    fonte = por_frame["fonte"].to_numpy().astype(object)

    # buracos longos: posição do modelo A onde houver previsão (dentro dos tempos de jogo)
    xy_modelo, prob_h = prever_buracos()
    usar = (fonte == "estimado") & np.isfinite(xy_modelo[:, 0])
    xy[usar] = xy_modelo[usar]
    prob_h = np.where(fonte == "estimado", prob_h, np.nan)

    # time mais próximo onde a bola é vista
    vista = np.isin(fonte, ["observado", "preenchido"])
    dh, da = _dist_por_time(xy, jog_m, eh_h)
    time_proximo = np.where(vista & (dh <= da), "H", np.where(vista & (da < dh), "A", ""))
    cod = np.where(time_proximo == "H", 1, np.where(time_proximo == "A", 0, -1))

    time_antes = np.full(n, "", dtype=object)
    time_depois = np.full(n, "", dtype=object)
    saiu = np.zeros(n, bool)
    posse = np.full(n, "", dtype=object)
    posse_modelo = np.full(n, "", dtype=object)
    letra = {1: "H", 0: "A", -1: ""}
    for _, b in buracos.iterrows():
        a, z = int(b.inicio), int(b.fim)
        if z - a < MIN_BURACO:
            continue
        antes = maioria(cod[max(0, a - JANELA_PONTA):a])
        depois = maioria(cod[z:z + JANELA_PONTA])
        time_antes[a:z], time_depois[a:z] = letra[antes], letra[depois]
        saiu[a:z] = bool(b.sumiu_fora_do_campo)
        if saiu[a]:
            posse[a:z] = posse_modelo[a:z] = "D"
            continue
        seguro = antes if antes >= 0 else depois
        posse[a:z] = letra[seguro]
        p = prob_h[a:z]
        p = np.where(np.isfinite(p), p, 0.5)
        posse_modelo[a:z] = [letra[int(c)] for c in rotular(antes, depois, z - a, "modelo", p)]

    return pd.DataFrame({
        "frame": np.arange(n), "x": xy[:, 0], "y": xy[:, 1], "fonte": fonte,
        "buraco_id": por_frame["buraco_id"].to_numpy(), "time_proximo": time_proximo,
        "dist_h": dh, "dist_a": da, "time_antes": time_antes, "time_depois": time_depois,
        "saiu_campo": saiu, "prob_h": prob_h, "posse_buraco": posse,
        "posse_buraco_modelo": posse_modelo,
    })


def carregar_sinal(regerar=False):
    if regerar or not os.path.isfile(SAIDA):
        sinal = gerar()
        os.makedirs(os.path.dirname(SAIDA), exist_ok=True)
        sinal.to_csv(SAIDA, index=False, float_format="%.3f")
        return sinal
    return pd.read_csv(SAIDA, keep_default_na=False, na_values=[""],
                       dtype={"time_proximo": str, "time_antes": str, "time_depois": str,
                              "posse_buraco": str, "posse_buraco_modelo": str})


def rotulos_exemplo(sinal, coluna_buraco="posse_buraco", janela=50):
    """Como a frente 4 pode usar o sinal: time mais próximo onde a bola é vista, a sugestão do
    buraco onde ela some, e a mesma suavização ±2 s do baseline. Só para conferência."""
    from starter.baseline import smooth
    t = sinal["time_proximo"].fillna("").to_numpy()
    s = sinal[coluna_buraco].fillna("").to_numpy()
    letra = np.where(t != "", t, s)
    cod = np.where(letra == "H", 0, np.where(letra == "A", 1, -1)).astype(np.int8)
    return smooth(cod, janela)


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(ROOT, "evaluation"))
    import possession_metrics as pm

    sinal = carregar_sinal(regerar=True)
    print("salvo em", SAIDA)
    g = np.zeros(len(sinal), bool)
    g[11185:90652] = True
    g[110808:190198] = True
    print("\nfonte nos frames avaliados:")
    print(sinal.loc[g, "fonte"].value_counts(normalize=True).map("{:.1%}".format).to_string())

    gt = pm.label_codes(pd.read_csv(os.path.join(ROOT, "evaluation", "gt",
                                                 "BAR-ATH_possession_gt_public.csv")))
    print("\nconferência no trecho público (17 posses, só ordem de grandeza):")
    for col in ("posse_buraco", "posse_buraco_modelo"):
        r = pm.evaluate(gt, pm.label_codes(rotulos_exemplo(sinal, col)), bootstrap_n=0)
        print("  {:<20} spellF1 {:.4f}  macroF1 {:.4f}".format(col, r["spell_f1_mean"], r["macro_f1"]))
