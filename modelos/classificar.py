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
Saida: p_fake de cada modelo e do stacking (colunas p_fake_<modelo>, p_fake_stacking).
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


def classificar(textos, pasta, truncar=100):
    modelos, meta = carregar(pasta)
    X = np.array([limpar_noticia(t, truncar) for t in textos], dtype=object)
    saida = pd.DataFrame({"n_palavras_usadas": [len(x.split()) for x in X]})
    for n in meta["ordem"]:
        saida[f"p_fake_{n}"] = modelos[n].predict_proba(X)[:, 1]
    Z = np.column_stack([logit(saida[f"p_fake_{n}"].to_numpy()) for n in meta["ordem"]])
    saida["p_fake_stacking"] = meta["meta"].predict_proba(Z)[:, 1]
    saida["classe_stacking"] = np.where(saida["p_fake_stacking"] >= 0.5, "Fake", "Real")
    return saida


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--texto")
    g.add_argument("--arquivo", help="txt com UMA noticia")
    g.add_argument("--csv")
    ap.add_argument("--coluna", default="Noticia")
    ap.add_argument("--base", default="FakeRecogna2_extrativa_casada")
    ap.add_argument("--truncar", type=int, default=100)
    ap.add_argument("--saida", default=None, help="CSV de saida (opcional)")
    a = ap.parse_args()

    pasta = RAIZ / "modelos_salvos" / a.base
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
    res = classificar(textos, pasta, a.truncar)
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