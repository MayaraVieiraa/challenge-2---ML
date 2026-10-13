# -*- coding: utf-8 -*-
"""
tratamento/preparar_fakebr.py

Converte o Fake.br Corpus (3.600 falsas + 3.600 verdadeiras, em pares) num CSV de TESTE EXTERNO,
com o mesmo tratamento do treino (minusculas, sem "none", corte em N palavras). Ele NUNCA entra no
treino: serve para medir se o modelo funciona numa colecao de outra origem e de outra epoca.

Antes (uma vez, FORA da pasta do projeto; sao ~21 mil arquivos):
    git clone --depth 1 https://github.com/roneysco/Fake.br-Corpus.git ~/Downloads/Fake.br-Corpus

Uso (raiz do projeto):
    python3 tratamento/preparar_fakebr.py --repo ~/Downloads/Fake.br-Corpus --truncar 100

Usa a pasta size_normalized_texts (em cada par, o texto maior e cortado ao tamanho do menor), o que
neutraliza o tamanho entre as classes. Saida: dados_externos/fakebr_t<N>.csv
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "modelos"))
from _inferencia import limpar_noticia  # noqa: E402

RE_DOM = re.compile(r"https?://(?:www\.)?([^/]+)")


def ler_meta(arq):
    """Linhas do *-meta.txt: autor, link, categoria, data, ..."""
    if not arq.exists():
        return {}
    L = [l.strip() for l in arq.read_text(encoding="utf-8", errors="ignore").splitlines()]
    L += [""] * 4
    m = RE_DOM.search(L[1])
    return {"Dominio": m.group(1) if m else "", "DataTexto": L[3]}


def ano(txt):
    m = re.search(r"(20\d{2})", txt or "")
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="pasta clonada do Fake.br-Corpus")
    ap.add_argument("--truncar", type=int, default=100, help="primeiras N palavras (padrao 100)")
    a = ap.parse_args()
    repo = Path(a.repo).expanduser()
    if not (repo / "size_normalized_texts").exists():
        sys.exit(f"Nao achei {repo}/size_normalized_texts. Confira --repo.")

    linhas = []
    for cls, rot in (("fake", "Fake"), ("true", "Real")):
        arqs = sorted((repo / "size_normalized_texts" / cls).glob("*.txt"), key=lambda p: int(p.stem))
        for f in arqs:
            bruto = f.read_text(encoding="utf-8", errors="ignore")
            meta = ler_meta(repo / "full_texts" / f"{cls}-meta-information" / f"{f.stem}-meta.txt")
            linhas.append({"id": f"{cls}_{f.stem}", "Noticia": limpar_noticia(bruto, a.truncar), "Rotulo": rot,
                           "Dominio": meta.get("Dominio", ""), "Ano": ano(meta.get("DataTexto"))})
    d = pd.DataFrame(linhas)
    d["n_palavras"] = d["Noticia"].str.split().str.len()
    d = d[d["n_palavras"] >= 5].reset_index(drop=True)
    saida = RAIZ / "dados_externos"
    saida.mkdir(exist_ok=True)
    arq = saida / f"fakebr_t{a.truncar}.csv"
    d.to_csv(arq, index=False, encoding="utf-8-sig")

    print(f"Salvo: {arq}  ({len(d)} textos)")
    print("\nClasses:", d["Rotulo"].value_counts().to_dict())
    print("\nn_palavras por classe (deve ser parecido):")
    print(d.groupby("Rotulo")["n_palavras"].describe().round(1).to_string())
    print("\nAno por classe:")
    print(pd.crosstab(d["Ano"].fillna(0).astype(int), d["Rotulo"]).to_string())
    print("\nDominios (10 maiores por classe):")
    for rot in ("Fake", "Real"):
        print(f"  {rot}:", d[d["Rotulo"] == rot]["Dominio"].value_counts().head(10).to_dict())


if __name__ == "__main__":
    main()