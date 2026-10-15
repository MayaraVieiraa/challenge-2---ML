# -*- coding: utf-8 -*-
"""
modelos/classificar.py

Classifica noticias NOVAS (texto cru da web) com os modelos salvos por
`avaliar_externo.py --salvar` (treinados so no desenvolvimento da base casada).

AVISO: no teste externo (Fake.br) o AUC foi de ~0,72 a 0,78, bem abaixo dos ~0,98 da base.
Trate p_fake como um indicio, nao como veredito. Uma noticia verdadeira de outra epoca ou
genero pode receber p_fake alto.

Uso (raiz do projeto):
    python3 modelos/classificar.py --texto "cole aqui o texto da noticia"
    python3 modelos/classificar.py --arquivo noticia.txt
    python3 modelos/classificar.py --csv noticias.csv --coluna Noticia --saida resultado.csv

--truncar N (padrao 100): usa so as N primeiras palavras, como no treino/teste externo.
Saida: p_fake de cada modelo e do stacking (colunas p_fake_<modelo>, p_fake_stacking) e a classe
em TRES faixas: Real (p < --limiar-real), Fake (p >= --limiar-fake) e Incerto (entre os dois).

Por que tres faixas: no Fake.br, com limiar unico de 0,5, 26% das noticias REAIS sairam como Fake
(sobretudo politica de 2016-2018: a base de treino tem Real quase so de 2020-2021, sobre covid).
Treinando tambem com metade dos pares do Fake.br (avaliar_externo.py --treino-externo 0.5), nos pares de
teste: com Fake a partir de 0,7, 4,9% dos Reais saem como Fake e 81% dos Fake sao detectados; ~14% das
noticias caem em "Incerto" e devem ser revisadas por uma pessoa. Em fontes/epocas novas espere pior.
"""
import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
from _inferencia import limpar_noticia  # noqa: E402


def logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def carregar(pasta):
    meta = joblib.load(pasta / "stacking_meta.joblib")
    modelos = {n: joblib.load(pasta / f"{n}.joblib") for n in meta["ordem"]}
    return modelos, meta


LIMIAR_REAL = 0.3
LIMIAR_FAKE = 0.7


def faixa(p, limiar_real=LIMIAR_REAL, limiar_fake=LIMIAR_FAKE):
    return np.where(p >= limiar_fake, "Fake", np.where(p < limiar_real, "Real", "Incerto"))


def classificar(textos, pasta, truncar=100, limiar_real=LIMIAR_REAL, limiar_fake=LIMIAR_FAKE):
    modelos, meta = carregar(pasta)
    X = np.array([limpar_noticia(t, truncar) for t in textos], dtype=object)
    saida = pd.DataFrame({"n_palavras_usadas": [len(x.split()) for x in X]})
    for n in meta["ordem"]:
        saida[f"p_fake_{n}"] = modelos[n].predict_proba(X)[:, 1]
    Z = np.column_stack([logit(saida[f"p_fake_{n}"].to_numpy()) for n in meta["ordem"]])
    saida["p_fake_stacking"] = meta["meta"].predict_proba(Z)[:, 1]
    saida["classe_stacking"] = faixa(saida["p_fake_stacking"].to_numpy(), limiar_real, limiar_fake)
    return saida


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--texto")
    g.add_argument("--arquivo", help="txt com UMA noticia")
    g.add_argument("--csv")
    ap.add_argument("--coluna", default="Noticia")
    ap.add_argument("--base", default="FakeRecogna2_extrativa_casada")
    ap.add_argument("--somente-base", action="store_true", help="usa os modelos treinados so na FakeRecogna")
    ap.add_argument("--truncar", type=int, default=100)
    ap.add_argument("--saida", default=None, help="CSV de saida (opcional)")
    ap.add_argument("--limiar-real", type=float, default=LIMIAR_REAL, help="abaixo disso: Real (padrao 0,3)")
    ap.add_argument("--limiar-fake", type=float, default=LIMIAR_FAKE, help="a partir disso: Fake (padrao 0,7)")
    a = ap.parse_args()
    if not 0 <= a.limiar_real <= a.limiar_fake <= 1:
        sys.exit("Use 0 <= --limiar-real <= --limiar-fake <= 1")

    pasta = RAIZ / "modelos_salvos" / a.base
    if (RAIZ / "modelos_salvos" / f"{a.base}_com_fakebr" / "stacking_meta.joblib").exists() and not a.somente_base:
        pasta = RAIZ / "modelos_salvos" / f"{a.base}_com_fakebr"   # treinados tambem com metade do Fake.br
    if not (pasta / "stacking_meta.joblib").exists():
        sys.exit(f"Nao achei os modelos em {pasta}. Rode antes:\n"
                 "  python3 modelos/avaliar_externo.py --externo dados_externos/fakebr_t100.csv --salvar")
    if a.texto:
        textos = [a.texto]
    elif a.arquivo:
        textos = [Path(a.arquivo).read_text(encoding="utf-8")]
    else:
        df = pd.read_csv(a.csv)
        textos = df[a.coluna].fillna("").astype(str).tolist()
    if a.truncar and any(len(t.split()) < 20 for t in textos):
        print("[aviso] ha textos com menos de 20 palavras; o treino so tem textos de ~50 a 190 palavras.")
    res = classificar(textos, pasta, a.truncar, a.limiar_real, a.limiar_fake)
    if a.csv:
        res = pd.concat([df.reset_index(drop=True), res], axis=1)
    pd.set_option("display.width", 200)
    cols = [c for c in res.columns if c.startswith("p_fake") or c == "classe_stacking"]
    print(res[cols].round(3).to_string())
    if a.saida:
        res.to_csv(a.saida, index=False)
        print(f"\nSalvo em {a.saida}")


if __name__ == "__main__":
    main()