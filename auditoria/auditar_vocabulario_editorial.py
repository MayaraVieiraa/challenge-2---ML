# -*- coding: utf-8 -*-
"""
auditar_vocabulario_editorial.py — dependência de VOCABULÁRIO EDITORIAL
=======================================================================
Hipótese do protocolo (seção 6.3): grande parte da classe Fake são matérias de agências de
checagem que DESMENTEM boatos. O modelo pode estar aprendendo o JEITO de escrever uma checagem
("é falso", "circula nas redes", "checamos") em vez de sinais intrínsecos de falsidade.

Verificações (todas só no desenvolvimento, nos 5 folds do protocolo):
  1. Prevalência de um léxico editorial (veredito / checagem / circulação) por classe, no
     Titulo e na Noticia, e AUC de uma regra trivial "conta marcadores".
  2. Quanto do TOPO dos coeficientes da regressão logística é vocabulário editorial.
  3. Ablações do texto: sem léxico, sem veredito+checagem, só corpo sem léxico, só as primeiras
     / últimas 40 palavras (aberturas e fechamentos padronizados).
  4. Curva de degradação: AUC ao remover, DENTRO de cada fold, as K features mais fortes.
  5. Leave-one-source-out: treina sem uma fonte (Dominio, usado só para agrupar) e testa nela.
     Se o desempenho cai ao trocar de fonte, o modelo aprendeu o jeito da fonte.

Saídas em auditoria_outputs/: vocabulario_*.csv/png, vocabulario_editorial.md/json
Uso:  python auditoria/auditar_vocabulario_editorial.py [--rapido]
"""
import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _comum import (LEXICO_EDITORIAL, SEED, auc_ic95, carregar_df, categoria_editorial,  # noqa: E402
                    contar_marcadores, cv_oof, dividir, montar_texto, pasta_saida, pipeline_lr,
                    remover_lexico, resumo_metricas, salvar_json, salvar_md, tabela_md, tfidf_protocolo,
                    veredito)


# ------------------------------------------------------------------ 1. prevalência
def prevalencia(df):
    y = df["y"].to_numpy()
    linhas, scores = [], {}
    for campo in ("Titulo", "Noticia"):
        cont = contar_marcadores(df[campo].fillna("").astype(str))
        cont["qualquer (veredito+checagem+circulação)"] = cont["total"]
        cont["veredito ou checagem"] = cont["veredito"] + cont["checagem"]
        for cat in ("veredito", "checagem", "circulacao", "veredito ou checagem",
                    "qualquer (veredito+checagem+circulação)"):
            m = (cont[cat] > 0).to_numpy()
            pf, pr = float(m[y == 1].mean()), float(m[y == 0].mean())
            linhas.append({"campo": campo, "categoria": cat, "pct_docs_fake": pf, "pct_docs_real": pr,
                           "razao_fake_real": pf / pr if pr > 0 else np.inf,
                           "precisao_para_fake": float(y[m].mean()) if m.any() else np.nan,
                           "auc_contagem": float(roc_auc_score(y, cont[cat])) if cont[cat].nunique() > 1 else np.nan})
        scores[campo] = cont
    return pd.DataFrame(linhas), scores


# ------------------------------------------------------------------ 2. coeficientes
def top_coeficientes(Xd, yd, topn=200):
    m = pipeline_lr().fit(Xd, yd)
    termos = m.named_steps["tfidf"].get_feature_names_out()
    coef = m.named_steps["lr"].coef_.ravel()
    tab = pd.DataFrame({"termo": termos, "coef": coef})
    tab["lado"] = np.where(tab.coef > 0, "Fake", "Real")
    tab["abs_coef"] = tab.coef.abs()
    tab["categoria_editorial"] = [categoria_editorial(t) for t in tab.termo]
    tab = tab.sort_values("abs_coef", ascending=False).reset_index(drop=True)
    linhas = []
    for K in (20, 50, 100, 200):
        for lado in ("Fake", "Real"):
            t = tab[tab.lado == lado].head(K)
            ed = t.categoria_editorial != ""
            linhas.append({"top_K_por_lado": K, "lado": lado, "n_editoriais": int(ed.sum()),
                           "pct_termos_editoriais": float(ed.mean()),
                           "pct_do_peso_abs_editorial": float(t.loc[ed, "abs_coef"].sum() / t["abs_coef"].sum())})
    return tab.head(topn * 2), pd.DataFrame(linhas)


# ------------------------------------------------------------------ 3. ablações
def ablacoes(dev, yd, folds):
    base = montar_texto(dev, "titulo_noticia")
    noticia = montar_texto(dev, "noticia")
    limpo = montar_texto(dev, "titulo_limpo_noticia")
    pri = np.array([" ".join(t.split()[:40]) for t in noticia])
    ult = np.array([" ".join(t.split()[-40:]) for t in noticia])
    variantes = {
        "Titulo+Noticia (protocolo)": base,
        "sem léxico veredito+checagem": np.array([remover_lexico(t, ("veredito", "checagem")) for t in base]),
        "sem léxico completo (veredito+checagem+circulação)": np.array([remover_lexico(t) for t in base]),
        "Noticia só, sem léxico completo": np.array([remover_lexico(t) for t in noticia]),
        "Titulo limpo + Noticia, sem léxico completo": np.array([remover_lexico(t) for t in limpo]),
        "Noticia: só as primeiras 40 palavras": pri,
        "Noticia: só as últimas 40 palavras": ult,
    }
    linhas = []
    for nome, X in variantes.items():
        t0 = time.time()
        p = cv_oof(X, yd, folds, pipeline_lr)
        lo, hi = auc_ic95(yd, p)
        linhas.append({"variante": nome, **resumo_metricas(yd, p), "auc_ic95_inf": lo, "auc_ic95_sup": hi})
        print(f"  [{time.time() - t0:3.0f}s] {nome:52s} AUC={linhas[-1]['roc_auc']:.4f}")
    return pd.DataFrame(linhas)


# ------------------------------------------------------------------ 4. curva de degradação
def _curva_fold(k, X, y, folds, Ks):
    tr, va = folds != k, folds == k
    tf = tfidf_protocolo()
    Xtr, Xva = tf.fit_transform(X[tr]), tf.transform(X[va])
    lr = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED).fit(Xtr, y[tr])
    ordem = np.argsort(-np.abs(lr.coef_.ravel()))
    saida = {}
    for K in Ks:
        manter = np.ones(Xtr.shape[1], bool)
        manter[ordem[:K]] = False
        m = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED).fit(Xtr[:, manter], y[tr])
        saida[K] = (m.predict_proba(Xva[:, manter])[:, 1], va)
    return saida


def curva_degradacao(X, y, folds, Ks):
    res = Parallel(n_jobs=2)(delayed(_curva_fold)(k, X, y, folds, Ks) for k in range(5))
    linhas = []
    for K in Ks:
        oof = np.zeros(len(y))
        for r in res:
            p, va = r[K]
            oof[va] = p
        linhas.append({"K_features_removidas": K, "roc_auc": float(roc_auc_score(y, oof)),
                       "brier": resumo_metricas(y, oof)["brier"]})
    return pd.DataFrame(linhas)


# ------------------------------------------------------------------ 5. leave-one-source-out
def _loso(dom, X, y, doms, minimo):
    te = doms == dom
    m = pipeline_lr().fit(X[~te], y[~te])
    p = m.predict_proba(X[te])[:, 1]
    yt = y[te]
    r = {"dominio": dom, "n": int(te.sum()), "pct_fake_no_dominio": float(yt.mean())}
    if len(np.unique(yt)) == 2:
        r["roc_auc"] = float(roc_auc_score(yt, p))
    r["accuracy"] = float(((p >= 0.5) == yt).mean())
    r["p_fake_media"] = float(p.mean())
    return r


def leave_one_source_out(dev, yd, entrada, minimo=100):
    X = montar_texto(dev, entrada)
    doms = dev["Dominio"].fillna("?").to_numpy()
    lista = [d for d, c in pd.Series(doms).value_counts().items() if c >= minimo]
    res = Parallel(n_jobs=2)(delayed(_loso)(d, X, yd, doms, minimo) for d in lista)
    t = pd.DataFrame(res)
    t.insert(0, "entrada", entrada)
    return t


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--saida-base", default=None)
    ap.add_argument("--rapido", action="store_true", help="pula a curva de degradação e o leave-one-source-out")
    args = ap.parse_args()
    saida = pasta_saida(args.saida_base)

    df = carregar_df()
    y = df["y"].to_numpy()
    ids_dev, ids_te, folds = dividir(y)
    dev = df.iloc[ids_dev].reset_index(drop=True)
    yd = y[ids_dev]
    md, js = [], {}
    md += ["**Léxico editorial usado** (radicais, aplicados a texto sem acento; ver `auditoria/_comum.py`):", ""]
    md += [f"- `{k}`: `{v}`" for k, v in LEXICO_EDITORIAL.items()] + [""]

    print("1. prevalência do léxico...")
    prev, _ = prevalencia(df)
    prev.to_csv(saida / "vocabulario_1_prevalencia.csv", index=False, encoding="utf-8")
    q = prev[(prev.campo == "Noticia") & (prev.categoria == "veredito ou checagem")].iloc[0]
    qt = prev[(prev.campo == "Titulo") & (prev.categoria == "veredito ou checagem")].iloc[0]
    qa = prev[(prev.campo == "Noticia") & (prev.categoria.str.startswith("qualquer"))].iloc[0]
    md += ["### 1. Presença do léxico editorial por classe (base completa, contagem)", "", tabela_md(prev), "",
           veredito("ATENCAO" if q["razao_fake_real"] >= 3 else "INFO",
                    f"Noticia: {q['pct_docs_fake']:.1%} dos Fake x {q['pct_docs_real']:.1%} dos Real contêm vocabulário de "
                    f"veredito/checagem (razão {q['razao_fake_real']:.1f}x). Titulo: {qt['pct_docs_fake']:.1%} x "
                    f"{qt['pct_docs_real']:.1%}."),
           veredito("INFO", f"Uma regra trivial 'conta marcadores editoriais na Noticia' já tem AUC = {qa['auc_contagem']:.3f}."), ""]
    js["prevalencia_noticia_veredito_checagem"] = {"fake": q["pct_docs_fake"], "real": q["pct_docs_real"]}
    js["auc_regra_marcadores_noticia"] = qa["auc_contagem"]

    print("2. coeficientes...")
    topo, resumo_topo = top_coeficientes(montar_texto(dev, "titulo_noticia"), yd)
    topo.to_csv(saida / "vocabulario_2_top_termos.csv", index=False, encoding="utf-8")
    resumo_topo.to_csv(saida / "vocabulario_2_resumo_top_termos.csv", index=False, encoding="utf-8")
    t50 = resumo_topo[(resumo_topo.top_K_por_lado == 50) & (resumo_topo.lado == "Fake")].iloc[0]
    r50 = resumo_topo[(resumo_topo.top_K_por_lado == 50) & (resumo_topo.lado == "Real")].iloc[0]
    md += ["### 2. Vocabulário editorial no topo dos coeficientes (LR, Titulo+Noticia)", "", tabela_md(resumo_topo), "",
           "Termos mais fortes para **Fake**: " + ", ".join(topo[topo.lado == "Fake"].head(25).termo), "",
           "Termos mais fortes para **Real**: " + ", ".join(topo[topo.lado == "Real"].head(25).termo), "",
           veredito("ATENCAO" if t50["pct_termos_editoriais"] >= 0.3 else "INFO",
                    f"{t50['pct_termos_editoriais']:.0%} dos 50 maiores termos pró-Fake são editoriais "
                    f"(contra {r50['pct_termos_editoriais']:.0%} pró-Real)."), ""]
    js["top50_editorial_fake"] = t50["pct_termos_editoriais"]

    print("3. ablações do texto (5 folds cada)...")
    ab = ablacoes(dev, yd, folds)
    ab.to_csv(saida / "vocabulario_3_ablacoes.csv", index=False, encoding="utf-8")
    auc0, auc_lex, auc_cont = (float(ab.loc[0, "roc_auc"]), float(ab.loc[2, "roc_auc"]), float(ab.loc[3, "roc_auc"]))
    auc_pri, auc_ult = float(ab.loc[5, "roc_auc"]), float(ab.loc[6, "roc_auc"])
    md += ["### 3. Ablações do texto (LR + TF-IDF, 5 folds)", "", tabela_md(ab), "",
           veredito("ATENCAO" if auc_lex >= 0.95 else "OK",
                    f"removido TODO o léxico editorial a AUC vai de {auc0:.4f} para {auc_lex:.4f} (queda {auc0 - auc_lex:+.4f}); "
                    f"só corpo + sem léxico = {auc_cont:.4f}. "
                    + ("O sinal NÃO está só nas palavras do léxico: o modelo capta também estilo/estrutura/tema da checagem."
                       if auc_lex >= 0.95 else "O desempenho depende fortemente do léxico editorial.")),
           veredito("ATENCAO" if max(auc_pri, auc_ult) >= 0.9 else "INFO",
                    f"só as primeiras 40 palavras: AUC {auc_pri:.4f}; só as últimas 40: {auc_ult:.4f} — "
                    "aberturas e fechamentos padronizados são suficientes para separar as classes."), ""]
    js["ablacoes"] = dict(zip(ab["variante"], ab["roc_auc"]))

    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
    a = ab.iloc[::-1]
    axs[0].barh(a["variante"], a["roc_auc"], color="#2a6fdb",
                xerr=[a["roc_auc"] - a["auc_ic95_inf"], a["auc_ic95_sup"] - a["roc_auc"]])
    axs[0].set(xlim=(0.8, 1.0), xlabel="ROC-AUC (OOF, IC95%)", title="Ablações do vocabulário editorial")
    axs[0].tick_params(axis="y", labelsize=7)

    if not args.rapido:
        print("4. curva de degradação (remove as K features mais fortes, dentro do fold)...")
        Ks = [0, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
        c1 = curva_degradacao(montar_texto(dev, "titulo_noticia"), yd, folds, Ks)
        c2 = curva_degradacao(montar_texto(dev, "noticia"), yd, folds, Ks)
        c1["entrada"], c2["entrada"] = "Titulo+Noticia", "Noticia só"
        cur = pd.concat([c1, c2])
        cur.to_csv(saida / "vocabulario_4_curva_degradacao.csv", index=False, encoding="utf-8")
        for nome, c in (("Titulo+Noticia", c1), ("Noticia só", c2)):
            axs[1].plot(np.maximum(c.K_features_removidas, 1), c.roc_auc, "o-", label=nome)
        axs[1].set(xscale="log", xlabel="K features mais fortes removidas (escala log)", ylabel="ROC-AUC (OOF)",
                   title="Curva de degradação")
        axs[1].legend()
        piso = c1[c1.K_features_removidas == 500]["roc_auc"].iloc[0]
        md += ["### 4. Curva de degradação (remove as K features de maior |coeficiente| dentro de cada fold)", "",
               tabela_md(cur.pivot(index="K_features_removidas", columns="entrada", values="roc_auc").reset_index()), "",
               veredito("ATENCAO" if piso >= 0.95 else "INFO",
                        f"removendo as 500 features mais fortes a AUC ainda é {piso:.4f}: o sinal é DIFUSO (milhares de pistas "
                        "redundantes), não depende de meia dúzia de palavras-chave — o que dificulta 'consertar' o vazamento "
                        "apenas filtrando palavras."), ""]
        js["curva_degradacao_K500"] = float(piso)
    fig.tight_layout()
    fig.savefig(saida / "vocabulario_ablacoes.png", dpi=140)
    plt.close(fig)

    if not args.rapido:
        print("5. leave-one-source-out por domínio...")
        lo = pd.concat([leave_one_source_out(dev, yd, "titulo_noticia"), leave_one_source_out(dev, yd, "noticia")])
        lo.to_csv(saida / "vocabulario_5_leave_one_source_out.csv", index=False, encoding="utf-8")
        mist = lo[(lo.pct_fake_no_dominio > 0.05) & (lo.pct_fake_no_dominio < 0.95)]
        so_fake = lo[(lo.pct_fake_no_dominio >= 0.95) & (lo.entrada == "titulo_noticia")]
        so_real = lo[(lo.pct_fake_no_dominio <= 0.05) & (lo.entrada == "titulo_noticia")]
        md += ["### 5. Leave-one-source-out (treina sem o domínio, testa nele)", "",
               "`accuracy` em domínio de uma só classe = taxa de acerto dessa classe; `roc_auc` só existe em domínio misto.", "",
               tabela_md(lo.sort_values(["entrada", "n"], ascending=[True, False])), ""]
        if len(so_fake):
            md += [veredito("INFO", "domínios só-Fake deixados de fora (Titulo+Noticia): acerto médio "
                            f"{so_fake.accuracy.mean():.1%} (mín. {so_fake.accuracy.min():.1%} em {so_fake.loc[so_fake.accuracy.idxmin(), 'dominio']})."), ""]
        if len(mist):
            md += [veredito("INFO", "domínios mistos deixados de fora: AUC médio "
                            f"{mist[mist.entrada == 'titulo_noticia'].roc_auc.mean():.4f} (Titulo+Noticia) e "
                            f"{mist[mist.entrada == 'noticia'].roc_auc.mean():.4f} (Noticia só) — compare com a CV do protocolo."), ""]
        js["loso"] = lo.to_dict("records")

    salvar_md(saida, "vocabulario_editorial", "Auditoria de dependência de vocabulário editorial", md)
    salvar_json(saida, "vocabulario_editorial", js)
    print(f"\nRelatório: {saida / 'vocabulario_editorial.md'}")


if __name__ == "__main__":
    main()
