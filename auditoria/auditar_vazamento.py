# -*- coding: utf-8 -*-
"""
auditar_vazamento.py — auditoria de VAZAMENTO DE RÓTULO
=======================================================
Pergunta: o modelo está acertando por causa de informação que "entrega" o rótulo e que não
seria sinal intrínseco de falsidade? Seis verificações:

  A. Uso de colunas: quais colunas os scripts de modelagem realmente leem (análise estática)
     e se alguma é proibida pelo protocolo.
  B. Poder preditivo ISOLADO de cada coluna proibida (por que elas são proibidas) + Subtitulo.
  C. Marcadores de veredito escritos no Titulo ("#boato", "É #FAKE", "É falso"...).
  D. Ablação do texto de entrada: Titulo+Noticia x Noticia x Titulo limpo x só Titulo...
  E. Teste de sanidade do pipeline: rótulos EMBARALHADOS devem dar AUC ≈ 0,5; e comparação
     TF-IDF ajustado dentro do fold (correto) x em todo o desenvolvimento (vazado de propósito).
  F. Estrutura por ano / domínio / categoria e avaliação temporal (treina no passado,
     testa no futuro).

Só usa os 80% de desenvolvimento e os 5 folds do protocolo (o teste final não é tocado).
Saídas em auditoria_outputs/: vazamento_*.csv/png, vazamento.md, vazamento.json
Uso:  python auditoria/auditar_vazamento.py [--rapido]
"""
import argparse
import re
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _comum import (RAIZ, SEED, auc_ic95, carregar_df, cv_oof, dividir, limpar_titulo, montar_texto,
                    pasta_saida, pipeline_lr, resumo_metricas, salvar_json, salvar_md, tabela_md,
                    tfidf_protocolo, veredito)

PROIBIDAS = ["Categoria", "Dominio", "URL", "Autor", "Classe", "Rotulo", "Data", "Ano", "AnoMes",
             "Data_dt", "Subtitulo", "n_caracteres", "n_palavras", "n_palavras_unicas", "diversidade_lexica",
             "tam_medio_palavra", "n_palavras_titulo", "tem_subtitulo", "titulo_maiusculas_%",
             "n_exclamacoes_titulo", "n_interrogacoes_titulo"]
ESTRUTURAIS = ["n_caracteres", "n_palavras", "n_palavras_unicas", "diversidade_lexica", "tam_medio_palavra",
               "n_palavras_titulo", "tem_subtitulo", "titulo_maiusculas_%", "n_exclamacoes_titulo",
               "n_interrogacoes_titulo"]


# ------------------------------------------------------------------ A. uso de colunas
def auditar_uso_colunas():
    linhas = []
    arquivos = sorted((RAIZ / "modelos").glob("treinar_*.py")) + [RAIZ / "modelos" / "_entrada.py"]
    for arq in arquivos:
        src = arq.read_text(encoding="utf-8")
        # descarta docstrings e comentários, que citam colunas só para explicar
        src = re.sub(r'"""(.*?)"""', "", src, flags=re.S)
        src = re.sub(r"#.*", "", src)
        usadas = sorted(set(re.findall(r"""\[\s*["']([^"']+)["']\s*\]""", src)))
        usadas = [c for c in usadas if c in PROIBIDAS or c in ("Titulo", "Noticia")]
        proibidas = [c for c in usadas if c not in ("Titulo", "Noticia", "Rotulo")]
        linhas.append({"arquivo": arq.name, "colunas_lidas": ", ".join(usadas),
                       "proibidas_lidas": ", ".join(proibidas) or "—",
                       "status": "VIOLA" if proibidas else "ok"})
    return pd.DataFrame(linhas)


# ------------------------------------------------------------------ B. colunas proibidas isoladas
def _fab_cat():
    return Pipeline([("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
                     ("lr", LogisticRegression(max_iter=2000, random_state=SEED))])


def _fab_num():
    return HistGradientBoostingClassifier(max_depth=3, max_iter=100, random_state=SEED)


def poder_colunas_proibidas(dev, yd, folds):
    linhas = []

    def add(nome, tipo, p):
        linhas.append({"coluna": nome, "tipo": tipo, "roc_auc_isolada": roc_auc_score(yd, p)})

    for c in ["Categoria", "Dominio", "Autor", "Ano", "AnoMes", "tem_subtitulo"]:
        X = dev[[c]].fillna("__ausente__").astype(str).to_numpy()
        add(c, "categórica", cv_oof(X, yd, folds, _fab_cat))
    for c in ESTRUTURAIS:
        if c == "tem_subtitulo":
            continue
        add(c, "estrutural (numérica)", cv_oof(dev[[c]].fillna(0).to_numpy(float), yd, folds, _fab_num))
    add("10 estruturais juntas", "estrutural (numérica)",
        cv_oof(dev[ESTRUTURAIS].fillna(0).to_numpy(float), yd, folds, _fab_num))
    add("Subtitulo (texto TF-IDF)", "texto", cv_oof(dev["Subtitulo"].fillna("").to_numpy(str), yd, folds, pipeline_lr))
    return pd.DataFrame(linhas).sort_values("roc_auc_isolada", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------ C. marcadores no título
def marcadores_titulo(df):
    t = df["Titulo"].fillna("").astype(str)
    padroes = {
        "sufixo '#boato'": t.str.contains(r"#\s*boato", case=False, regex=True),
        "prefixo 'É #FAKE'": t.str.contains(r"^\s*é\s+#?\s*fake", case=False, regex=True),
        "prefixo 'É falso/enganoso/...'": t.str.contains(
            r"^\s*é\s+(?:falso|falsa|enganoso|enganosa|exagerado|impreciso|verdade|fato)\b", case=False, regex=True),
        "qualquer hashtag (#)": t.str.contains("#", regex=False),
        "palavra 'boato' no título": t.str.contains(r"\bboato", case=False, regex=True),
        "palavra 'fake' no título": t.str.contains(r"\bfake", case=False, regex=True),
        "título muda ao limpar (regra completa)": t.map(lambda s: limpar_titulo(s) != " ".join(s.split())),
    }
    y = df["y"].to_numpy()
    linhas = []
    for nome, m in padroes.items():
        m = m.to_numpy()
        nf, nr = int((m & (y == 1)).sum()), int((m & (y == 0)).sum())
        linhas.append({"marcador": nome, "docs_fake": nf, "docs_real": nr,
                       "cobertura_dos_fake": nf / max(int((y == 1).sum()), 1),
                       "precisao_para_fake": nf / max(nf + nr, 1)})
    return pd.DataFrame(linhas)


# ------------------------------------------------------------------ D. ablação da entrada
def ablacao_entrada(dev, yd, folds):
    variantes = {
        "Titulo + Noticia (PROTOCOLO)": montar_texto(dev, "titulo_noticia"),
        "Noticia só": montar_texto(dev, "noticia"),
        "Titulo limpo + Noticia": montar_texto(dev, "titulo_limpo_noticia"),
        "Titulo só": dev["Titulo"].fillna("").astype(str).to_numpy(),
        "Titulo limpo só": dev["Titulo"].fillna("").astype(str).map(limpar_titulo).to_numpy(),
    }
    linhas, oofs = [], {}
    for nome, X in variantes.items():
        t0 = time.time()
        p = cv_oof(X, yd, folds, pipeline_lr)
        oofs[nome] = p
        lo, hi = auc_ic95(yd, p)
        linhas.append({"entrada": nome, **resumo_metricas(yd, p), "auc_ic95_inf": lo, "auc_ic95_sup": hi})
        print(f"  [{time.time() - t0:3.0f}s] {nome:32s} AUC={linhas[-1]['roc_auc']:.4f}")
    return pd.DataFrame(linhas), oofs


# ------------------------------------------------------------------ E. sanidade do pipeline
def sanidade_pipeline(dev, yd, folds, n_perm=3):
    X = montar_texto(dev, "titulo_noticia")
    rng = np.random.default_rng(SEED)
    linhas = []
    for i in range(n_perm):
        yp = rng.permutation(yd)
        linhas.append({"teste": f"rótulos embaralhados #{i + 1}", "roc_auc": roc_auc_score(yp, cv_oof(X, yp, folds))})
    # TF-IDF "vazado" de propósito: vocabulário/IDF ajustados em TODO o desenvolvimento
    tf = tfidf_protocolo().fit(X)
    Xv = tf.transform(X)
    p_vaz = cv_oof(Xv, yd, folds, lambda: LogisticRegression(C=1.0, max_iter=2000, random_state=SEED))
    p_ok = cv_oof(X, yd, folds, pipeline_lr)
    linhas.append({"teste": "TF-IDF ajustado em todo o dev (vazado de propósito)", "roc_auc": roc_auc_score(yd, p_vaz)})
    linhas.append({"teste": "TF-IDF ajustado dentro do fold (protocolo)", "roc_auc": roc_auc_score(yd, p_ok)})
    return pd.DataFrame(linhas)


# ------------------------------------------------------------------ F. ano / domínio / categoria / tempo
def perfil_estrutura(df, dev, yd, saida):
    d = df.copy()
    d["Ano"] = d["Ano"].astype("Int64").astype(str)
    for col in ["Ano", "Dominio", "Categoria"]:
        t = d.groupby([col, "Rotulo"]).size().unstack(fill_value=0)
        for c in ("Fake", "Real"):
            if c not in t:
                t[c] = 0
        t["pct_fake"] = t["Fake"] / (t["Fake"] + t["Real"])
        t.to_csv(saida / f"vazamento_perfil_{col.lower()}.csv", encoding="utf-8")
    dom = pd.read_csv(saida / "vazamento_perfil_dominio.csv")
    unico = dom[(dom.pct_fake == 0) | (dom.pct_fake == 1)]
    pct_dom_exclusivo = float(unico[["Fake", "Real"]].to_numpy().sum() / dom[["Fake", "Real"]].to_numpy().sum())

    # avaliação temporal dentro do desenvolvimento
    dd = dev.copy()
    dd["dt"] = pd.to_datetime(dd["Data_dt"], errors="coerce")
    dd["yd"] = yd
    dd = dd.dropna(subset=["dt"])
    corte = dd["dt"].median()
    tr, te = dd[dd.dt < corte], dd[dd.dt >= corte]
    linhas = [{"corte": str(corte.date()), "treino_n": len(tr), "treino_pct_fake": float(tr.yd.mean()),
               "teste_n": len(te), "teste_pct_fake": float(te.yd.mean())}]
    for ent in ("titulo_noticia", "noticia"):
        try:
            m = pipeline_lr().fit(montar_texto(tr, ent), tr.yd.to_numpy())
            p = m.predict_proba(montar_texto(te, ent))[:, 1]
            linhas[0][f"auc_temporal_{ent}"] = float(roc_auc_score(te.yd, p)) if te.yd.nunique() == 2 else np.nan
            linhas[0][f"recall_fake_temporal_{ent}"] = float((p[te.yd.to_numpy() == 1] >= 0.5).mean())
            linhas[0][f"recall_real_temporal_{ent}"] = float((p[te.yd.to_numpy() == 0] < 0.5).mean())
        except Exception as e:  # treino com uma classe só, etc.
            linhas[0][f"auc_temporal_{ent}"] = np.nan
            linhas[0][f"erro_{ent}"] = str(e)[:80]
    return pd.DataFrame(linhas), pct_dom_exclusivo


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--saida-base", default=None)
    ap.add_argument("--rapido", action="store_true", help="pula o teste de permutação (3 execuções extras)")
    args = ap.parse_args()
    saida = pasta_saida(args.saida_base)

    df = carregar_df()
    y = df["y"].to_numpy()
    ids_dev, ids_te, folds = dividir(y)
    dev = df.iloc[ids_dev].reset_index(drop=True)
    yd = y[ids_dev]
    md, js = [], {}

    print("A. uso de colunas nos scripts...")
    a = auditar_uso_colunas()
    a.to_csv(saida / "vazamento_A_uso_colunas.csv", index=False, encoding="utf-8")
    viola = a[a.status == "VIOLA"]
    md += ["### A. Colunas lidas pelos scripts de modelagem", "", tabela_md(a), "",
           veredito("CRITICO", f"scripts leem colunas proibidas: {', '.join(viola.arquivo)}") if len(viola) else
           veredito("OK", "nenhum script de modelagem lê coluna proibida (só Titulo, Noticia e o rótulo)."), ""]

    print("B. poder preditivo das colunas proibidas...")
    b = poder_colunas_proibidas(dev, yd, folds)
    b.to_csv(saida / "vazamento_B_colunas_proibidas.csv", index=False, encoding="utf-8")
    md += ["### B. Quanto cada coluna proibida 'entregaria' o rótulo (ROC-AUC OOF, 1 coluna por vez)", "",
           tabela_md(b), "",
           veredito("INFO", "estas colunas NÃO entram nos modelos; a tabela justifica a proibição. "
                    f"Maior AUC isolada: {b.iloc[0]['coluna']} = {b.iloc[0]['roc_auc_isolada']:.3f}."), ""]
    js["colunas_proibidas_auc"] = dict(zip(b["coluna"], b["roc_auc_isolada"]))

    print("C. marcadores no título...")
    c = marcadores_titulo(df)
    c.to_csv(saida / "vazamento_C_marcadores_titulo.csv", index=False, encoding="utf-8")
    regra = c.iloc[-1]
    nivel = "CRITICO" if regra["cobertura_dos_fake"] >= 0.30 else ("ATENCAO" if regra["cobertura_dos_fake"] >= 0.05 else "OK")
    md += ["### C. Marcadores de veredito escritos no Titulo (base completa, só contagem)", "", tabela_md(c), "",
           veredito(nivel, f"{regra['cobertura_dos_fake']:.1%} dos títulos Fake trazem uma marca de veredito "
                    f"(precisão para Fake = {regra['precisao_para_fake']:.1%}). Usar Titulo na entrada "
                    "equivale a dar parte do rótulo ao modelo."), ""]
    js["titulo_marca"] = {"cobertura_fake": regra["cobertura_dos_fake"], "precisao_fake": regra["precisao_para_fake"]}

    print("D. ablação do texto de entrada (5 folds cada)...")
    d, oofs = ablacao_entrada(dev, yd, folds)
    d.to_csv(saida / "vazamento_D_ablacao_entrada.csv", index=False, encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    dd = d.iloc[::-1]
    ax.barh(dd["entrada"], dd["roc_auc"], color="#2a6fdb",
            xerr=[dd["roc_auc"] - dd["auc_ic95_inf"], dd["auc_ic95_sup"] - dd["roc_auc"]])
    ax.axvline(0.90, c="gray", ls="--", lw=1)
    ax.text(0.9, -0.55, " meta 0,90", fontsize=8, color="gray")
    ax.set(xlim=(min(0.85, d["roc_auc"].min() - 0.02), 1.0), xlabel="ROC-AUC (OOF, IC95%)",
           title="Ablação do texto de entrada — LR + TF-IDF")
    fig.tight_layout()
    fig.savefig(saida / "vazamento_D_ablacao_entrada.png", dpi=140)
    plt.close(fig)
    auc_proto = float(d.loc[0, "roc_auc"])
    auc_not = float(d.loc[1, "roc_auc"])
    auc_limpo = float(d.loc[2, "roc_auc"])
    auc_tit_limpo = float(d.loc[4, "roc_auc"])
    md += ["### D. Ablação do texto de entrada (LR + TF-IDF, 5 folds do protocolo)", "", tabela_md(d), "",
           veredito("ATENCAO" if auc_proto - auc_limpo >= 0.002 else "INFO",
                    f"Titulo+Noticia = {auc_proto:.4f}; Noticia só = {auc_not:.4f}; Titulo limpo+Noticia = {auc_limpo:.4f}. "
                    f"O título sozinho (já limpo das marcas óbvias) ainda alcança {auc_tit_limpo:.4f}."),
           veredito("ATENCAO" if auc_not >= 0.95 else "OK",
                    f"mesmo SEM título a AUC é {auc_not:.4f}: o sinal está também no corpo do texto "
                    "(ver auditar_vocabulario_editorial.py)."), ""]
    js["ablacao_entrada"] = dict(zip(d["entrada"], d["roc_auc"]))

    print("E. sanidade do pipeline...")
    if args.rapido:
        e = pd.DataFrame()
    else:
        e = sanidade_pipeline(dev, yd, folds)
        e.to_csv(saida / "vazamento_E_sanidade_pipeline.csv", index=False, encoding="utf-8")
        perm = e[e.teste.str.startswith("rótulos")]["roc_auc"]
        ok = perm.between(0.47, 0.53).all()
        vaz, cor = e.iloc[-2]["roc_auc"], e.iloc[-1]["roc_auc"]
        md += ["### E. Sanidade do pipeline", "", tabela_md(e), "",
               veredito("OK" if ok else "CRITICO",
                        f"AUC com rótulos embaralhados = {perm.mean():.3f} (esperado ≈ 0,50): "
                        + ("o pipeline não cria sinal do nada." if ok else "há vazamento no pipeline!")),
               veredito("OK" if abs(vaz - cor) < 0.002 else "ATENCAO",
                        f"TF-IDF vazado x correto: {vaz:.4f} x {cor:.4f} (diferença {vaz - cor:+.4f}); "
                        "ajustar o vetorizador fora do fold quase não muda o resultado aqui."), ""]
        js["permutacao_auc_media"] = float(perm.mean())

    print("F. ano / domínio / categoria / avaliação temporal...")
    f, pct_dom = perfil_estrutura(df, dev, yd, saida)
    f.to_csv(saida / "vazamento_F_avaliacao_temporal.csv", index=False, encoding="utf-8")
    t_ano = pd.read_csv(saida / "vazamento_perfil_ano.csv", dtype={"Ano": str})
    md += ["### F. Estrutura por ano, domínio e categoria; avaliação temporal", "",
           "Fake por ano:", "", tabela_md(t_ano), "",
           f"Domínios que publicam UMA só classe cobrem {pct_dom:.1%} da base. Perfis completos em "
           "`vazamento_perfil_dominio.csv` e `vazamento_perfil_categoria.csv`.", "",
           "Avaliação temporal (treina antes da data mediana do desenvolvimento, testa depois):", "", tabela_md(f), "",
           veredito("ATENCAO", "a classe Fake está concentrada em 2019–2020 e quase não existe antes; a divisão temporal "
                    "mistura efeito de TEMPO com efeito de FONTE, então não é uma medida limpa de generalização."), ""]
    js["pct_base_em_dominio_exclusivo"] = pct_dom

    salvar_md(saida, "vazamento", "Auditoria de vazamento de rótulo", md)
    salvar_json(saida, "vazamento", js)
    print(f"\nRelatório: {saida / 'vazamento.md'}")


if __name__ == "__main__":
    main()
