# -*- coding: utf-8 -*-
"""
tratamento/casar_tamanho.py

Remove o atalho do TAMANHO do texto. Na FakeRecogna 2.0 as noticias reais foram resumidas (quase
todas com 77 a 132 palavras) e as falsas nao (de 1 a mais de 2000), entao so o numero de palavras
separa as classes com AUC ~0,88. Aqui o conjunto e CASADO por faixa de tamanho: em cada faixa de
LARGURA palavras ficam quantidades IGUAIS de Fake e Real (sorteio com seed 42 na classe maior).
Faixas em que uma das classes tem menos de MIN_POR_CLASSE linhas sao descartadas.

Depois mede de novo: o AUC usando SO o tamanho deve cair para perto de 0,50.

Uso (raiz do projeto, depois de preparar_fakerecogna2.py):
    python3 tratamento/casar_tamanho.py --variante extrativa
Saidas em dados_tratados/:
    FakeRecogna2_<variante>_casada.csv
    FakeRecogna2_<variante>_casada_relatorio.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

RAIZ = Path(__file__).resolve().parent.parent
SEED = 42


def auc_so_tamanho(d):
    y = (d["Rotulo"] == "Fake").astype(int).to_numpy()
    X = d[["n_palavras", "n_caracteres"]].to_numpy()
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    a = cross_val_score(DecisionTreeClassifier(max_depth=4, random_state=0), X, y, cv=cv, scoring="roc_auc")
    return float(a.mean()), float(a.std())


def faixas(n_palavras, largura, teto):
    """Faixas de `largura` palavras ate `teto`; tudo acima de `teto` vira uma faixa so."""
    return np.minimum(np.asarray(n_palavras) // largura, teto // largura).astype(int)


def casar(d, largura, teto, min_por_classe):
    d = d.copy()
    d["faixa"] = faixas(d["n_palavras"], largura, teto)
    rng = np.random.default_rng(SEED)
    partes, descartadas = [], []
    for f, g in d.groupby("faixa"):
        fk, rl = g[g["Rotulo"] == "Fake"], g[g["Rotulo"] == "Real"]
        n = min(len(fk), len(rl))
        if n < min_por_classe:
            descartadas.append({"faixa_inicio": int(f * largura), "fake": len(fk), "real": len(rl)})
            continue
        partes.append(fk.iloc[rng.permutation(len(fk))[:n]])
        partes.append(rl.iloc[rng.permutation(len(rl))[:n]])
    out = pd.concat(partes).drop(columns="faixa").sort_index()
    return out.reset_index(drop=True), descartadas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variante", default="extrativa")
    ap.add_argument("--largura", type=int, default=10, help="largura da faixa, em palavras (padrao 10)")
    ap.add_argument("--teto", type=int, default=200, help="acima disso tudo cai na ultima faixa (padrao 200)")
    ap.add_argument("--min-por-classe", type=int, default=20, help="minimo por classe em cada faixa (padrao 20)")
    a = ap.parse_args()

    entrada = RAIZ / "dados_tratados" / f"FakeRecogna2_{a.variante}_dedup.csv"
    if not entrada.exists():
        sys.exit(f"Nao achei {entrada}. Rode antes preparar_fakerecogna2.py")
    d = pd.read_csv(entrada)
    auc0 = auc_so_tamanho(d)
    c, descartadas = casar(d, a.largura, a.teto, a.min_por_classe)
    auc1 = auc_so_tamanho(c)

    rel = {"variante": a.variante, "largura_faixa": a.largura, "teto": a.teto, "min_por_classe": a.min_por_classe,
           "linhas_antes": int(len(d)), "linhas_depois": int(len(c)),
           "por_classe_antes": d["Rotulo"].value_counts().to_dict(),
           "por_classe_depois": c["Rotulo"].value_counts().to_dict(),
           "auc_so_tamanho_antes": {"media": round(auc0[0], 3), "dp": round(auc0[1], 3)},
           "auc_so_tamanho_depois": {"media": round(auc1[0], 3), "dp": round(auc1[1], 3)},
           "mediana_palavras_depois": c.groupby("Rotulo")["n_palavras"].median().to_dict(),
           "faixas_descartadas": descartadas,
           "grupos_url_repetida_depois": int(c.loc[c["Grupo"].duplicated(keep=False), "Grupo"].nunique())
           if "Grupo" in c.columns else None}
    c.to_csv(RAIZ / "dados_tratados" / f"FakeRecogna2_{a.variante}_casada.csv", index=False, encoding="utf-8-sig")
    (RAIZ / "dados_tratados" / f"FakeRecogna2_{a.variante}_casada_relatorio.json").write_text(
        json.dumps(rel, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rel, ensure_ascii=False, indent=2))
    print(f"\nAUC so com o tamanho: {auc0[0]:.3f} -> {auc1[0]:.3f}  (0,50 = nenhum sinal)")
    print(f"Linhas: {len(d)} -> {len(c)}")
    if auc1[0] > 0.60:
        print("[ALERTA] Ainda sobra sinal de tamanho (AUC > 0,60). Tente --largura 5 ou um --teto menor.")
    if len(c) < 10000:
        print("[ALERTA] Sobraram poucas linhas (< 10 mil). Considere --largura maior.")


if __name__ == "__main__":
    main()