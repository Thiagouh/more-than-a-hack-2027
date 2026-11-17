"""Modelo A: aprende, nos frames em que a bola foi vista, a estimar a bola a partir dos jogadores.

Como os buracos reais não têm resposta, o treino usa buracos sintéticos: apagamos trechos
observados com as mesmas durações dos buracos reais, calculamos as features exatamente como
seriam calculadas num buraco real (pontas do buraco + jogadores) e usamos a posição verdadeira
como alvo. Dois alvos:
  - posição (x, y) em metros (regressão);
  - time do jogador mais próximo da bola (classificação), que é o que interessa para a posse.

Validação cruzada por tempo: treina num tempo de jogo e testa no outro, para não vazar
informação entre frames vizinhos. Nenhum gabarito é usado.

    python -m src.bola.modelo
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.bola.carregar import carregar, em_metros  # noqa: E402
from src.bola.estimar import mean_shift, palpite_temporal  # noqa: E402
from src.bola.limpar import limpar, marcar_falhas, trechos  # noqa: E402

# Params
SEMENTE = 2027
N_BURACOS_TREINO = 4000   # buracos sintéticos sorteados
PASSO = 5                 # usa 1 a cada 5 frames de cada buraco (frames vizinhos são quase iguais)
CONTEXTO = 6              # frames observados exigidos antes do buraco (velocidade de saída)
RAIO_CONTAGEM = 10.0      # m
TEMPOS = {1: (11185, 90652), 2: (110808, 190198)}


def velocidades(jog_m, passo=5, fps=25):
    v = np.full_like(jog_m, np.nan)
    v[passo:] = (jog_m[passo:] - jog_m[:-passo]) * fps / passo
    return v


def features(bola, a, z, frames, jog_m, jog_v, eh_h):
    """Matriz de features para os frames `frames` de um buraco [a, z).

    `bola` só é lida nas pontas do buraco (a-1, a-CONTEXTO e z), como num buraco real.
    """
    L = z - a
    pal = palpite_temporal(bola, a, z)
    rel = frames - a
    p = pal[rel]
    antes, depois = bola[a - 1], bola[z] if z < len(bola) else bola[a - 1]
    v_saida = (bola[a - 1] - bola[a - CONTEXTO]) * 25 / (CONTEXTO - 1) \
        if np.isfinite(bola[a - CONTEXTO, 0]) else np.array([np.nan, np.nan])
    P = jog_m[frames]
    V = jog_v[frames]
    ms = mean_shift(p, P, sigma=8, peso=0.0)
    n = len(frames)
    cols = {
        "t": rel / L, "dur": np.full(n, L / 25), "falta": (L - rel) / 25, "passou": rel / 25,
        "antes_x": np.full(n, antes[0]), "antes_y": np.full(n, antes[1]),
        "depois_x": np.full(n, depois[0]), "depois_y": np.full(n, depois[1]),
        "vsaida_x": np.full(n, v_saida[0]), "vsaida_y": np.full(n, v_saida[1]),
        "interp_x": p[:, 0], "interp_y": p[:, 1], "ms_x": ms[:, 0], "ms_y": ms[:, 1],
    }
    with np.errstate(all="ignore"), _sem_avisos():
        for nome, m in (("H", eh_h), ("A", ~eh_h)):
            Pt, Vt = P[:, m], V[:, m]
            cols["c{}_x".format(nome)] = np.nanmean(Pt[..., 0], 1)
            cols["c{}_y".format(nome)] = np.nanmean(Pt[..., 1], 1)
            cols["s{}_x".format(nome)] = np.nanstd(Pt[..., 0], 1)
            cols["s{}_y".format(nome)] = np.nanstd(Pt[..., 1], 1)
            cols["v{}_x".format(nome)] = np.nanmean(Vt[..., 0], 1)
            cols["v{}_y".format(nome)] = np.nanmean(Vt[..., 1], 1)
            for ref, q in (("i", p), ("m", ms)):
                d = np.linalg.norm(Pt - q[:, None], axis=2)
                cols["d{}{}".format(nome, ref)] = np.nanmin(d, 1)
                cols["n{}{}".format(nome, ref)] = (d < RAIO_CONTAGEM).sum(1)
            # quanto o time está correndo em direção ao palpite
            dirp = q[:, None] - Pt
            norm = np.linalg.norm(dirp, axis=2)
            cols["conv{}".format(nome)] = np.nanmean((Vt * dirp).sum(2) / np.maximum(norm, 1), 1)
    return pd.DataFrame(cols)


class _sem_avisos:
    def __enter__(self):
        import warnings
        self._c = warnings.catch_warnings()
        self._c.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *a):
        self._c.__exit__(*a)


def time_mais_proximo(bola, P, eh_h):
    d = np.linalg.norm(P - bola[:, None], axis=2)
    d = np.where(np.isfinite(d), d, np.inf)
    return eh_h[d.argmin(1)].astype(int)  # 1 = H, 0 = A


def espelhar_2o_tempo(xy, pitch):
    """Gira o campo 180° a partir do intervalo, para o Barça atacar sempre para o mesmo lado.

    Os times trocam de lado no intervalo; sem isso, um modelo treinado num tempo aprende a
    geometria invertida do outro. A operação é a própria inversa.
    """
    out = xy.copy()
    out[TEMPOS[1][1]:] = np.asarray(pitch) - out[TEMPOS[1][1]:]
    return out


def preparar():
    dados = carregar()
    pitch = dados["pitch"]
    bola = em_metros(dados["bola"][:, :2]).astype(np.float64)
    bola[marcar_falhas(bola)] = np.nan
    bola = espelhar_2o_tempo(bola, pitch)
    jog_m = espelhar_2o_tempo(em_metros(dados["jogadores"][..., :2]).astype(np.float64), pitch)
    return dados, bola, jog_m, velocidades(jog_m), dados["times"] == "H"


def duracoes_reais(bola):
    obs = np.isfinite(bola[:, 0])
    L = []
    for t0, t1 in TEMPOS.values():
        a, z = trechos(~obs[t0:t1])
        L += list(z - a)
    L = np.array(L)
    return L[L > 12]


def sortear(bola, rng, n):
    """[(a, z, tempo)] de buracos sintéticos em trechos totalmente observados."""
    obs = np.isfinite(bola[:, 0])
    cs = np.r_[0, np.cumsum(obs)]
    durs = rng.choice(duracoes_reais(bola), size=n)
    out = []
    for L in durs:
        larg = L + CONTEXTO + 1
        tempo = int(rng.integers(1, 3))
        t0, t1 = TEMPOS[tempo]
        ok = np.flatnonzero(cs[larg:] - cs[:-larg] == larg)
        ok = ok[(ok >= t0) & (ok + larg <= t1)]
        if len(ok) == 0:
            continue
        s = int(rng.choice(ok))
        a = s + CONTEXTO
        out.append((a, a + int(L), tempo))
    return out


def amostras(bola, jog_m, jog_v, eh_h, buracos):
    Xs, ys, meta = [], [], []
    for k, (a, z, tempo) in enumerate(buracos):
        frames = np.arange(a, z, PASSO)
        Xs.append(features(bola, a, z, frames, jog_m, jog_v, eh_h))
        ys.append(np.c_[bola[frames], time_mais_proximo(bola[frames], jog_m[frames], eh_h)])
        meta.append(np.c_[np.full(len(frames), k), np.full(len(frames), tempo),
                          np.full(len(frames), z - a), frames,
                          np.full(len(frames), time_mais_proximo(bola[[a - 1]], jog_m[[a - 1]], eh_h)[0])])
    return (pd.concat(Xs, ignore_index=True), np.vstack(ys),
            pd.DataFrame(np.vstack(meta), columns=["buraco", "tempo", "dur", "frame", "time_antes"]))


class Modelo:
    """Prevê o desvio da bola em relação à interpolação (resíduo) e o time mais próximo.

    Prever o resíduo, e não a posição absoluta, deixa o modelo partir da interpolação, que já é
    boa perto das pontas do buraco, e aprender só a correção que os jogadores indicam.
    """

    def __init__(self):
        kw = dict(max_iter=400, learning_rate=0.05, random_state=SEMENTE)
        self.mx = HistGradientBoostingRegressor(**kw)
        self.my = HistGradientBoostingRegressor(**kw)
        self.mt = HistGradientBoostingClassifier(**kw)

    def fit(self, X, y):
        self.mx.fit(X, y[:, 0] - X.interp_x)
        self.my.fit(X, y[:, 1] - X.interp_y)
        self.mt.fit(X, y[:, 2].astype(int))
        return self

    def posicao(self, X):
        return np.c_[X.interp_x + self.mx.predict(X), X.interp_y + self.my.predict(X)]

    def prob_h(self, X):
        return self.mt.predict_proba(X)[:, 1]


def treinar(X, y):
    return Modelo().fit(X, y)


def validar():
    _, bola, jog_m, jog_v, eh_h = preparar()
    rng = np.random.default_rng(SEMENTE)
    X, y, meta = amostras(bola, jog_m, jog_v, eh_h, sortear(bola, rng, N_BURACOS_TREINO))
    print("amostras: {} frames de {} buracos sintéticos".format(len(X), meta.buraco.nunique()))

    pred = np.zeros((len(X), 3))
    for teste in (1, 2):
        tr, te = meta.tempo != teste, meta.tempo == teste
        m = treinar(X[tr], y[tr])
        pred[te, :2] = m.posicao(X[te])
        pred[te, 2] = m.prob_h(X[te])

    erro = {
        "interpolação": np.hypot(X.interp_x - y[:, 0], X.interp_y - y[:, 1]),
        "jogadores (regra)": np.hypot(0.5 * X.interp_x + 0.5 * X.ms_x - y[:, 0],
                                      0.5 * X.interp_y + 0.5 * X.ms_y - y[:, 1]),
        "modelo A": np.hypot(pred[:, 0] - y[:, 0], pred[:, 1] - y[:, 1]),
    }
    time_real = y[:, 2].astype(int)
    interp_xy = X[["interp_x", "interp_y"]].to_numpy()
    acerto = {
        "manter time de antes do buraco": meta.time_antes.to_numpy() == time_real,
        "mais próximo da interpolação": time_mais_proximo(interp_xy, jog_m[meta.frame.astype(int)], eh_h)
        == time_real,
        "modelo A (classificador)": (pred[:, 2] > 0.5) == time_real,
    }
    faixas = [(13, 250, "≤10 s"), (251, 500, "10-20 s"), (501, 10 ** 9, ">20 s")]

    print("\nErro de POSIÇÃO (mediana, m) - treino num tempo, teste no outro")
    linhas = []
    for nome, e in erro.items():
        linha = {"método": nome}
        for lo, hi, rot in faixas:
            m = (meta.dur >= lo) & (meta.dur <= hi)
            linha[rot] = np.median(e[m])
        linha["todos"] = np.median(e)
        linhas.append(linha)
    print(pd.DataFrame(linhas).to_string(index=False, float_format="%.1f"))

    print("\nAcerto do TIME mais próximo da bola (o que importa para a posse)")
    linhas = []
    for nome, a in acerto.items():
        linha = {"método": nome}
        for lo, hi, rot in faixas:
            m = ((meta.dur >= lo) & (meta.dur <= hi)).to_numpy()
            linha[rot] = a[m].mean()
        linha["todos"] = a.mean()
        linhas.append(linha)
    print(pd.DataFrame(linhas).to_string(index=False, float_format="%.3f"))

    conf = np.maximum(pred[:, 2], 1 - pred[:, 2])
    print("\nClassificador por nível de confiança:")
    for lim in (0.6, 0.7, 0.8, 0.9):
        m = conf >= lim
        print("  confiança ≥ {:.1f}: {:5.1%} dos frames, acerto {:.3f}".format(
            lim, m.mean(), acerto["modelo A (classificador)"][m].mean()))


if __name__ == "__main__":
    validar()


def prever_buracos():
    """Treina com todos os buracos sintéticos e aplica nos buracos reais dos dois tempos.

    Devolve (xy, prob_h): arrays por frame do vídeo, com NaN fora dos buracos. xy em metros, no
    sistema de coordenadas original (sem espelhamento).
    """
    dados, bola, jog_m, jog_v, eh_h = preparar()
    rng = np.random.default_rng(SEMENTE)
    X, y, _ = amostras(bola, jog_m, jog_v, eh_h, sortear(bola, rng, N_BURACOS_TREINO))
    modelo = treinar(X, y)

    xy = np.full((len(bola), 2), np.nan)
    prob_h = np.full(len(bola), np.nan)
    obs = np.isfinite(bola[:, 0])
    for t0, t1 in TEMPOS.values():
        for a, z in zip(*trechos(~obs[t0:t1])):
            a, z = a + t0, z + t0
            frames = np.arange(a, z)
            Xb = features(bola, a, z, frames, jog_m, jog_v, eh_h)
            xy[a:z] = modelo.posicao(Xb)
            prob_h[a:z] = modelo.prob_h(Xb)
    return espelhar_2o_tempo(xy, dados["pitch"]), prob_h
