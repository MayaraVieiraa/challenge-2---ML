# -*- coding: utf-8 -*-
"""
analisar_erros.py — análise de erros do baseline (protocolo, seção 7.7)
======================================================================
Amostra estratificada de 50 exemplos por quadrante, usando as previsões fora-do-fold do baseline
(modelos_outputs/baseline/oof_baseline.csv):
    FP  = Real previsto Fake (p >= 0,5)          FN = Fake previsto Real (p < 0,5)
    TP_alta = Fake com p >= 0,90                 TN_alta = Real com p <= 0,10
Gera uma planilha para CODIFICAÇÃO MANUAL (colunas vazias: tema, vocabulario, tamanho, estilo,
nomes, estrutura, periodo, ambiguidade) e um resumo automático por quadrante. Colunas de
metadados (Dominio, Categoria, Ano) aparecem só para ajudar a interpretar — não são features.

Saídas em modelos_outputs/baseline/: analise_erros_amostra.csv, analise_erros_resumo.txt
Uso:  python modelos/analisar_erros.py
"""
import os
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _entrada import ENTRADAS  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
DADOS = RAIZ / "dados_tratados" / os.environ.get("BASE_DADOS", "FakeRecogna2_extrativa_dedup.csv")
SEED = 42
N = 50
CATEGORIAS = ["tema", "vocabulario", "tamanho", "estilo", "nomes", "estrutura", "periodo", "ambiguidade"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada", choices=ENTRADAS, default="noticia")
    ap.add_argument("--saida-base", default=None)
    a = ap.parse_args()
    pasta = Path(a.saida_base or RAIZ / "modelos_outputs") / "baseline"
    oof = pd.read_csv(pasta / "oof_baseline.csv")
    df = pd.read_csv(DADOS)
    d = oof.merge(df.assign(id=np.arange(len(df))), on="id")
    p, y = d["p_fake_baseline"], d["y_fake"]
    quad = {"FP": (y == 0) & (p >= 0.5), "FN": (y == 1) & (p < 0.5),
            "TP_alta_confianca": (y == 1) & (p >= 0.9), "TN_alta_confianca": (y == 0) & (p <= 0.1)}
    partes, resumo = [], []
    for nome, m in quad.items():
        s = d[m]
        amostra = s.sample(min(N, len(s)), random_state=SEED).copy()
        amostra.insert(0, "quadrante", nome)
        partes.append(amostra)
        resumo.append(f"[{nome}] total no OOF = {len(s)} | amostrados = {len(amostra)} | "
                      f"n_palavras mediana = {s['n_palavras'].median():.0f} | p_fake média = {s['p_fake_baseline'].mean():.3f}\n"
                      f"   domínios: {s['Dominio'].value_counts().head(4).to_dict()}\n"
                      f"   categorias: {s['Categoria'].value_counts().head(4).to_dict()}\n"
                      f"   anos: {s['Ano'].value_counts().head(4).to_dict()}")
    out = pd.concat(partes)
    out["Titulo"] = out["Titulo"].fillna("")
    out["Noticia_inicio"] = out["Noticia"].fillna("").str.slice(0, 500)
    cols = ["quadrante", "id", "fold", "y_fake", "p_fake_baseline", "Titulo", "Noticia_inicio", "Dominio", "Categoria",
            "Ano", "n_palavras"]
    out = out[cols]
    for c in CATEGORIAS + ["observacao"]:
        out[c] = ""
    out.to_csv(pasta / "analise_erros_amostra.csv", index=False, encoding="utf-8-sig")
    (pasta / "analise_erros_resumo.txt").write_text("\n\n".join(resumo) + "\n", encoding="utf-8")
    print("\n\n".join(resumo))
    print(f"\nPlanilha para codificação manual: {pasta / 'analise_erros_amostra.csv'}")


if __name__ == "__main__":
    main()
