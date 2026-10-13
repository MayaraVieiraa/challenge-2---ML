# -*- coding: utf-8 -*-
"""
tratamento/checar_fakerecogna2.py

Duas checagens sobre a FakeRecogna 2.0, para rodar DEPOIS de preparar_fakerecogna2.py:

  1. URLs repetidas: o que sao? Sao o MESMO texto repetido (duplicata de verdade) ou textos
     DIFERENTES com a mesma URL? Isso decide se vale remover essas linhas.
  2. Tamanho como atalho: um classificador que ve SO o tamanho do texto (n. de palavras e de
     caracteres) consegue separar Fake de Real? Se o AUC for alto, o modelo pode acertar pelo
     tamanho, o que nao vale para noticias novas da web.

Uso (raiz do projeto):
    python3 tratamento/checar_fakerecogna2.py --variante extrativa
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

RAIZ = Path(__file__).resolve().parent.parent
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 110)


def url_valida(serie):
    s = serie.astype("string").str.strip()
    return s.mask(s.str.lower().isin(["", "null", "none", "nan", "<na>"]))


def checar_urls(variante):
    arq = RAIZ / "dados" / f"fakerecogna2_{variante}.csv"
    if not arq.exists():
        sys.exit(f"Nao achei {arq}. Rode antes o preparar_fakerecogna2.py")
    d = pd.read_csv(arq)
    d["URL"] = url_valida(d["URL"])
    # quem e Fake: mesma regra do script de tratamento (fontes de checagem)
    pad = (r"boatos\.org|e-farsas\.com|checamos\.afp\.com|projetocomprova\.com\.br|aosfatos\.org|"
           r"lupa\.uol|/fato-ou-fake|uol\.com\.br/confere|estadao-verifica")
    chec = d["URL"].astype("string").str.contains(pad, case=False, regex=True, na=False)
    fake_val = 1 if d.loc[chec, "Label"].mean() >= 0.5 else 0
    d["Rotulo"] = np.where(d["Label"] == fake_val, "Fake", "Real")

    com_url = d[d["URL"].notna()]
    rep = com_url[com_url["URL"].duplicated(keep=False)]
    print("=" * 78)
    print("1) URLs REPETIDAS")
    print("=" * 78)
    print(f"Linhas com URL preenchida: {len(com_url)} de {len(d)}")
    print(f"Linhas que fazem parte de um grupo com URL repetida: {len(rep)} "
          f"({rep['URL'].nunique()} URLs distintas)")
    print("Por classe:", rep["Rotulo"].value_counts().to_dict())
    if rep.empty:
        return
    g = rep.groupby("URL")["Noticia"].nunique()
    print(f"\nGrupos com texto IDENTICO (duplicata de verdade): {(g == 1).sum()}")
    print(f"Grupos com texto DIFERENTE (mesma URL, conteudos distintos): {(g > 1).sum()}")
    tam = rep.groupby("URL").size()
    print("\nTamanho dos grupos (quantas linhas por URL):", tam.value_counts().sort_index().to_dict())
    print("\nURLs mais repetidas:")
    print(rep["URL"].value_counts().head(8).to_string())
    dif = g[g > 1].index[:3]
    for u in dif:
        print(f"\n--- exemplo de URL com textos diferentes: {u}")
        for _, r in rep[rep["URL"] == u].iterrows():
            print(f"   [{r['Rotulo']}] {str(r['Noticia'])[:150]!r}")
    print("\nLEITURA: se predominar 'texto IDENTICO', remover e correto. Se predominar "
          "'texto DIFERENTE', NAO remova por URL: use apenas Noticia repetida e, na divisao "
          "treino/teste, mantenha a mesma URL do mesmo lado (GroupKFold).")


def checar_tamanho(variante):
    arq = RAIZ / "dados_tratados" / f"FakeRecogna2_{variante}_dedup.csv"
    if not arq.exists():
        sys.exit(f"Nao achei {arq}.")
    t = pd.read_csv(arq, usecols=["Rotulo", "n_palavras", "n_caracteres"])
    y = (t["Rotulo"] == "Fake").astype(int).to_numpy()
    X = t[["n_palavras", "n_caracteres"]].to_numpy()
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    # arvore pequena: captura "faixa de tamanho" (ex.: Real concentrado entre 77 e 132 palavras)
    auc = cross_val_score(DecisionTreeClassifier(max_depth=4, random_state=0), X, y, cv=cv,
                          scoring="roc_auc")
    print("\n" + "=" * 78)
    print("2) O TAMANHO SOZINHO SEPARA AS CLASSES?")
    print("=" * 78)
    print(f"AUC usando SO n_palavras e n_caracteres (arvore, 5 folds): {auc.mean():.3f} +- {auc.std():.3f}")
    print("Referencia: 0,50 = nenhum sinal; acima de ~0,75 o tamanho ja e um atalho forte.")
    print("\nTamanho (n_palavras) por faixa e classe, em % de cada classe:")
    bins = [0, 25, 50, 75, 100, 125, 150, 10**6]
    nomes = ["<=25", "26-50", "51-75", "76-100", "101-125", "126-150", ">150"]
    t["faixa"] = pd.cut(t["n_palavras"], bins=bins, labels=nomes)
    tab = (pd.crosstab(t["faixa"], t["Rotulo"], normalize="columns") * 100).round(1)
    print(tab.to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variante", default="extrativa")
    a = ap.parse_args()
    checar_urls(a.variante)
    checar_tamanho(a.variante)


if __name__ == "__main__":
    main()