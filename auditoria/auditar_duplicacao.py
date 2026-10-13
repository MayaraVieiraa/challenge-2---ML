# -*- coding: utf-8 -*-
"""
auditar_duplicacao.py — auditoria de DUPLICAÇÃO e QUASE-DUPLICAÇÃO
===================================================================
O corpus já foi deduplicado por URL e por texto exato (tratamento/auditar_duplicatas.py). Esta
auditoria vai além, porque o que infla a validação são cópias que a deduplicação exata NÃO vê:

  1. Duplicatas exatas residuais (URL, título, notícia, e versões normalizadas sem acento/pontuação),
     inclusive grupos que cruzam folds ou o teste, e conflitos de rótulo.
  2. QUASE-duplicatas por similaridade de cosseno em TF-IDF (limiares 0,80 / 0,90 / 0,95):
     pares, documentos envolvidos, pares com rótulos conflitantes.
  3. Vazamento entre divisões: quanto dos documentos de VALIDAÇÃO (cada fold) e de TESTE têm
     uma quase-cópia no TREINO — e se a cópia tem o mesmo rótulo.
  4. Impacto no desempenho: AUC do baseline nos documentos COM x SEM quase-cópia no treino, e CV
     AGRUPADA (StratifiedGroupKFold, grupos = componentes de quase-duplicatas) x CV do protocolo.
  5. Títulos repetidos entre treino e validação.

Só usa texto e rótulos; o desempenho é medido nos 5 folds do desenvolvimento (teste final intocado).
Saídas em auditoria_outputs/: duplicacao_*.csv, duplicacao.md, duplicacao.json
Uso:  python auditoria/auditar_duplicacao.py [--entrada titulo_noticia|noticia|titulo_limpo_noticia]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _comum import (ENTRADAS, SEED, carregar_df, cv_oof, dividir, montar_texto, normalizar,  # noqa: E402
                    pasta_saida, pipeline_lr, salvar_json, salvar_md, tabela_md, veredito)

LIMIARES = (0.80, 0.90, 0.95)


# ------------------------------------------------------------------ 1. exatas
def duplicatas_exatas(df, grupo):
    """grupo: 0-4 = fold do desenvolvimento, 5 = teste."""
    y = df["y"].to_numpy()
    chaves = {
        "URL": df["URL"].fillna("").astype(str),
        "Titulo (bruto)": df["Titulo"].fillna("").astype(str).str.strip(),
        "Titulo (normalizado)": df["Titulo"].fillna("").map(normalizar),
        "Noticia (bruta)": df["Noticia"].fillna("").astype(str).str.strip(),
        "Noticia (normalizada)": df["Noticia"].fillna("").map(normalizar),
        "Titulo+Noticia (normalizado)": (df["Titulo"].fillna("") + " " + df["Noticia"].fillna("")).map(normalizar),
    }
    linhas = []
    for nome, k in chaves.items():
        k = k.where(k != "", other=pd.Series(np.arange(len(k)).astype(str), index=k.index))  # vazio não agrupa
        g = pd.DataFrame({"k": k, "y": y, "grp": grupo})
        tam = g.groupby("k")["y"].transform("size")
        rep = g[tam > 1]
        gg = rep.groupby("k").agg(n=("y", "size"), n_rot=("y", "nunique"), n_grp=("grp", "nunique"))
        linhas.append({"chave": nome, "grupos_repetidos": int(len(gg)),
                       "docs_em_grupos_repetidos": int(len(rep)),
                       "docs_excedentes": int((gg["n"] - 1).sum()),
                       "grupos_com_rotulos_conflitantes": int((gg["n_rot"] > 1).sum()),
                       "grupos_que_cruzam_folds_ou_teste": int((gg["n_grp"] > 1).sum())})
    return pd.DataFrame(linhas)


# ------------------------------------------------------------------ 2/3. quase-duplicatas
def similaridades(textos, grupo, isdev, y, df, limiar_pares=min(LIMIARES), tam=800):
    """Para cada doc: vizinho mais próximo NO TREINO (dev de outro fold; para o teste, todo o dev).
    Também coleta todos os pares com cosseno >= limiar_pares."""
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True, strip_accents="unicode",
                          dtype=np.float32)
    X = vec.fit_transform(textos).tocsr()
    n = X.shape[0]
    XT = X.T.tocsc()
    viz_sim = np.full(n, -1.0, dtype=np.float32)
    viz_id = np.full(n, -1, dtype=int)
    glob_sim = np.full(n, -1.0, dtype=np.float32)
    glob_id = np.full(n, -1, dtype=int)
    pares = []
    for ini in range(0, n, tam):
        fim = min(ini + tam, n)
        S = (X[ini:fim] @ XT).toarray()
        idx = np.arange(ini, fim)
        S[np.arange(fim - ini), idx] = -1.0                       # sem auto-similaridade
        gj = S.argmax(axis=1)
        glob_sim[ini:fim], glob_id[ini:fim] = S[np.arange(fim - ini), gj], gj
        valido = isdev[None, :] & (grupo[None, :] != grupo[idx][:, None])
        Sv = np.where(valido, S, -1.0)
        vj = Sv.argmax(axis=1)
        viz_sim[ini:fim], viz_id[ini:fim] = Sv[np.arange(fim - ini), vj], vj
        ii, jj = np.nonzero((S >= limiar_pares) & (np.arange(n)[None, :] > idx[:, None]))
        for a, b in zip(ii + ini, jj):
            pares.append((a, b, float(S[a - ini, b])))
    pares = pd.DataFrame(pares, columns=["id_a", "id_b", "sim"]).sort_values("sim", ascending=False)
    return viz_sim, viz_id, glob_sim, glob_id, pares


def resumo_limiares(pares, y, grupo, n, viz_sim, viz_id, isdev):
    linhas = []
    for lim in LIMIARES:
        p = pares[pares.sim >= lim]
        docs = pd.unique(np.concatenate([p.id_a.to_numpy(), p.id_b.to_numpy()])) if len(p) else np.array([], int)
        conf = int((y[p.id_a.to_numpy()] != y[p.id_b.to_numpy()]).sum()) if len(p) else 0
        cruza = int((grupo[p.id_a.to_numpy()] != grupo[p.id_b.to_numpy()]).sum()) if len(p) else 0
        dev_val = isdev & (viz_sim >= lim)
        te = (~isdev) & (viz_sim >= lim)
        concorda = float((y[viz_id[dev_val | te]] == y[dev_val | te]).mean()) if (dev_val | te).any() else np.nan
        linhas.append({"limiar_cosseno": lim, "pares": len(p), "docs_envolvidos": len(docs),
                       "pct_corpus": len(docs) / n, "pares_rotulo_conflitante": conf,
                       "pares_entre_folds_ou_teste": cruza,
                       "pct_validacao_com_quase_copia_no_treino": float(dev_val.sum() / isdev.sum()),
                       "pct_teste_com_quase_copia_no_dev": float(te.sum() / (~isdev).sum()),
                       "rotulo_igual_ao_da_copia": concorda})
    return pd.DataFrame(linhas)


# ------------------------------------------------------------------ 4. impacto
def _ajusta(tr, va, X, y):
    m = pipeline_lr().fit(X[tr], y[tr])
    return va, m.predict_proba(X[va])[:, 1]


def cv_agrupada(Xd, yd, pares, ids_dev, lim=0.80, n_jobs=2):
    pos = {g: i for i, g in enumerate(ids_dev)}
    p = pares[(pares.sim >= lim) & pares.id_a.isin(list(pos)) & pares.id_b.isin(list(pos))]
    a = p.id_a.map(pos).to_numpy()
    b = p.id_b.map(pos).to_numpy()
    n = len(ids_dev)
    g = coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
    ncomp, grupos = connected_components(g, directed=False)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(sgkf.split(Xd, yd, grupos))
    res = Parallel(n_jobs=n_jobs)(delayed(_ajusta)(tr, va, Xd, yd) for tr, va in splits)
    oof = np.zeros(n)
    for va, pr in res:
        oof[va] = pr
    return oof, ncomp, int(pd.Series(grupos).value_counts().max())


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada", choices=ENTRADAS, default="noticia")
    ap.add_argument("--saida-base", default=None)
    args = ap.parse_args()
    saida = pasta_saida(args.saida_base)

    df = carregar_df()
    y = df["y"].to_numpy()
    n = len(df)
    ids_dev, ids_te, folds = dividir(y)
    grupo = np.full(n, 5)
    grupo[ids_dev] = folds
    isdev = np.zeros(n, bool)
    isdev[ids_dev] = True
    textos = montar_texto(df, args.entrada)
    md, js = [], {}

    print("1. duplicatas exatas...")
    ex = duplicatas_exatas(df, grupo)
    ex.to_csv(saida / "duplicacao_1_exatas.csv", index=False, encoding="utf-8")
    md += ["### 1. Duplicatas exatas residuais (base já deduplicada por URL e texto)", "", tabela_md(ex), ""]
    exi = ex.set_index("chave")
    conf = int(exi.loc[[c for c in exi.index if c.startswith(("Noticia", "Titulo+", "URL"))],
                       "grupos_com_rotulos_conflitantes"].sum())
    md += [veredito("CRITICO" if conf else "OK",
                    f"{conf} grupo(s) de TEXTO/URL idêntico com rótulos conflitantes." if conf else
                    "nenhum texto/URL idêntico com rótulos conflitantes (o corpus é consistente)."), ""]
    t_rep = exi.loc["Titulo (normalizado)"]
    if t_rep["grupos_com_rotulos_conflitantes"]:
        md += [veredito("INFO", f"{int(t_rep['grupos_com_rotulos_conflitantes'])} título(s) idêntico(s) aparecem com rótulos "
                        "diferentes (textos distintos sob o mesmo título genérico) — não é duplicação do conteúdo."), ""]
    md += [veredito("ATENCAO" if t_rep["docs_excedentes"] > 0 else "OK",
                    f"{int(t_rep['docs_excedentes'])} títulos repetidos (normalizados) permanecem na base — o protocolo "
                    "deduplicou por URL e texto, não por título; "
                    f"{int(t_rep['grupos_que_cruzam_folds_ou_teste'])} desses grupos cruzam folds/teste."), ""]

    print("2/3. quase-duplicatas (cosseno TF-IDF)...")
    t0 = time.time()
    viz_sim, viz_id, glob_sim, glob_id, pares = similaridades(textos, grupo, isdev, y, df)
    print(f"   {len(pares)} pares com cosseno >= {min(LIMIARES)} ({time.time() - t0:.0f}s)")
    resumo = resumo_limiares(pares, y, grupo, n, viz_sim, viz_id, isdev)
    resumo.to_csv(saida / "duplicacao_2_quase_duplicatas.csv", index=False, encoding="utf-8")
    top = pares.head(3000).copy()
    for lado in ("a", "b"):
        ids = top[f"id_{lado}"].to_numpy()
        top[f"y_{lado}"] = y[ids]
        top[f"dominio_{lado}"] = df["Dominio"].to_numpy()[ids]
        top[f"split_{lado}"] = np.where(grupo[ids] == 5, "teste", "fold" + pd.Series(grupo[ids]).astype(str))
        top[f"titulo_{lado}"] = df["Titulo"].fillna("").str.slice(0, 90).to_numpy()[ids]
    top.to_csv(saida / "duplicacao_pares_quase_duplicados.csv", index=False, encoding="utf-8")
    md += ["### 2. Quase-duplicatas e vazamento entre divisões (cosseno TF-IDF, entrada = " + args.entrada + ")", "",
           tabela_md(resumo), ""]
    r90 = resumo[resumo.limiar_cosseno == 0.90].iloc[0]
    nivel = "ATENCAO" if r90["pct_validacao_com_quase_copia_no_treino"] >= 0.03 else "OK"
    md += [veredito(nivel, f"{r90['pct_validacao_com_quase_copia_no_treino']:.1%} dos documentos de validação têm uma "
                    f"quase-cópia (cos ≥ 0,90) no treino; em {r90['rotulo_igual_ao_da_copia']:.1%} dos casos a cópia tem o "
                    "mesmo rótulo."), ""]
    js["quase_duplicatas"] = resumo.to_dict("records")

    print("4. impacto no desempenho (CV padrão x CV agrupada)...")
    Xd, yd = textos[ids_dev], y[ids_dev]
    oof = cv_oof(Xd, yd, folds, pipeline_lr)
    auc_padrao = roc_auc_score(yd, oof)
    sim_dev = viz_sim[ids_dev]
    linhas = []
    for lim in LIMIARES:
        com = sim_dev >= lim
        linha = {"subconjunto": f"validação COM quase-cópia no treino (cos ≥ {lim:.2f})", "n": int(com.sum())}
        linha["roc_auc"] = float(roc_auc_score(yd[com], oof[com])) if com.sum() > 30 and len(np.unique(yd[com])) == 2 else np.nan
        linha["accuracy"] = float(((oof[com] >= 0.5) == yd[com]).mean()) if com.any() else np.nan
        linhas.append(linha)
        sem = ~com
        linhas.append({"subconjunto": f"validação SEM quase-cópia no treino (cos < {lim:.2f})", "n": int(sem.sum()),
                       "roc_auc": float(roc_auc_score(yd[sem], oof[sem])),
                       "accuracy": float(((oof[sem] >= 0.5) == yd[sem]).mean())})
    linhas.append({"subconjunto": "todos (CV do protocolo)", "n": len(yd), "roc_auc": float(auc_padrao),
                   "accuracy": float(((oof >= 0.5) == yd).mean())})
    oof_g, ncomp, maior = cv_agrupada(Xd, yd, pares, ids_dev, lim=0.80)
    auc_g = float(roc_auc_score(yd, oof_g))
    linhas.append({"subconjunto": f"CV AGRUPADA por componentes de quase-duplicatas (cos ≥ 0,80; {ncomp} grupos, maior={maior})",
                   "n": len(yd), "roc_auc": auc_g, "accuracy": float(((oof_g >= 0.5) == yd).mean())})
    imp = pd.DataFrame(linhas)
    imp.to_csv(saida / "duplicacao_4_impacto.csv", index=False, encoding="utf-8")
    delta = auc_padrao - auc_g
    md += ["### 3. Impacto no desempenho (LR + TF-IDF)", "", tabela_md(imp), "",
           veredito("OK" if delta < 0.003 else "ATENCAO",
                    f"AUC do protocolo {auc_padrao:.4f} x AUC com CV agrupada {auc_g:.4f} (queda de {delta:+.4f}). "
                    + ("A quase-duplicação não infla a validação de forma relevante." if delta < 0.003 else
                       "A validação está inflada por quase-cópias entre folds: reporte também a CV agrupada.")), ""]
    js["auc_protocolo"], js["auc_cv_agrupada"] = float(auc_padrao), auc_g

    print("5. títulos repetidos entre treino e validação...")
    tnorm = df["Titulo"].fillna("").map(normalizar).to_numpy()
    rep_treino = np.zeros(n, bool)
    concorda = np.full(n, np.nan)
    for k in range(6):
        alvo = np.where(grupo == k)[0]
        treino = np.where(isdev & (grupo != k))[0]
        mapa = pd.Series(y[treino], index=tnorm[treino]).groupby(level=0).mean()
        for i in alvo:
            if tnorm[i] and tnorm[i] in mapa.index:
                rep_treino[i] = True
                concorda[i] = float(round(mapa[tnorm[i]]) == y[i])
    t = pd.DataFrame([{"conjunto": "validação (dev, 5 folds)", "n": int(isdev.sum()),
                       "com_titulo_igual_no_treino": int(rep_treino[isdev].sum()),
                       "pct": float(rep_treino[isdev].mean()),
                       "rotulo_igual": float(np.nanmean(concorda[isdev]))},
                      {"conjunto": "teste", "n": int((~isdev).sum()),
                       "com_titulo_igual_no_treino": int(rep_treino[~isdev].sum()),
                       "pct": float(rep_treino[~isdev].mean()),
                       "rotulo_igual": float(np.nanmean(concorda[~isdev]))}])
    t.to_csv(saida / "duplicacao_5_titulos.csv", index=False, encoding="utf-8")
    md += ["### 4. Títulos repetidos entre treino e validação/teste", "", tabela_md(t), ""]

    salvar_md(saida, "duplicacao", "Auditoria de duplicação e quase-duplicação", md)
    salvar_json(saida, "duplicacao", js)
    print(f"\nRelatório: {saida / 'duplicacao.md'}")


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
auditar_duplicacao.py — auditoria de DUPLICAÇÃO e QUASE-DUPLICAÇÃO
===================================================================
O corpus já foi deduplicado por URL e por texto exato (tratamento/auditar_duplicatas.py). Esta
auditoria vai além, porque o que infla a validação são cópias que a deduplicação exata NÃO vê:

  1. Duplicatas exatas residuais (URL, título, notícia, e versões normalizadas sem acento/pontuação),
     inclusive grupos que cruzam folds ou o teste, e conflitos de rótulo.
  2. QUASE-duplicatas por similaridade de cosseno em TF-IDF (limiares 0,80 / 0,90 / 0,95):
     pares, documentos envolvidos, pares com rótulos conflitantes.
  3. Vazamento entre divisões: quanto dos documentos de VALIDAÇÃO (cada fold) e de TESTE têm
     uma quase-cópia no TREINO — e se a cópia tem o mesmo rótulo.
  4. Impacto no desempenho: AUC do baseline nos documentos COM x SEM quase-cópia no treino, e CV
     AGRUPADA (StratifiedGroupKFold, grupos = componentes de quase-duplicatas) x CV do protocolo.
  5. Títulos repetidos entre treino e validação.

Só usa texto e rótulos; o desempenho é medido nos 5 folds do desenvolvimento (teste final intocado).
Saídas em auditoria_outputs/: duplicacao_*.csv, duplicacao.md, duplicacao.json
Uso:  python auditoria/auditar_duplicacao.py [--entrada titulo_noticia|noticia|titulo_limpo_noticia]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _comum import (ENTRADAS, SEED, carregar_df, cv_oof, dividir, montar_texto, normalizar,  # noqa: E402
                    pasta_saida, pipeline_lr, salvar_json, salvar_md, tabela_md, veredito)

LIMIARES = (0.80, 0.90, 0.95)


# ------------------------------------------------------------------ 1. exatas
def duplicatas_exatas(df, grupo):
    """grupo: 0-4 = fold do desenvolvimento, 5 = teste."""
    y = df["y"].to_numpy()
    chaves = {
        "URL": df["URL"].fillna("").astype(str),
        "Titulo (bruto)": df["Titulo"].fillna("").astype(str).str.strip(),
        "Titulo (normalizado)": df["Titulo"].fillna("").map(normalizar),
        "Noticia (bruta)": df["Noticia"].fillna("").astype(str).str.strip(),
        "Noticia (normalizada)": df["Noticia"].fillna("").map(normalizar),
        "Titulo+Noticia (normalizado)": (df["Titulo"].fillna("") + " " + df["Noticia"].fillna("")).map(normalizar),
    }
    linhas = []
    for nome, k in chaves.items():
        k = k.where(k != "", other=pd.Series(np.arange(len(k)).astype(str), index=k.index))  # vazio não agrupa
        g = pd.DataFrame({"k": k, "y": y, "grp": grupo})
        tam = g.groupby("k")["y"].transform("size")
        rep = g[tam > 1]
        gg = rep.groupby("k").agg(n=("y", "size"), n_rot=("y", "nunique"), n_grp=("grp", "nunique"))
        linhas.append({"chave": nome, "grupos_repetidos": int(len(gg)),
                       "docs_em_grupos_repetidos": int(len(rep)),
                       "docs_excedentes": int((gg["n"] - 1).sum()),
                       "grupos_com_rotulos_conflitantes": int((gg["n_rot"] > 1).sum()),
                       "grupos_que_cruzam_folds_ou_teste": int((gg["n_grp"] > 1).sum())})
    return pd.DataFrame(linhas)


# ------------------------------------------------------------------ 2/3. quase-duplicatas
def similaridades(textos, grupo, isdev, y, df, limiar_pares=min(LIMIARES), tam=800):
    """Para cada doc: vizinho mais próximo NO TREINO (dev de outro fold; para o teste, todo o dev).
    Também coleta todos os pares com cosseno >= limiar_pares."""
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True, strip_accents="unicode",
                          dtype=np.float32)
    X = vec.fit_transform(textos).tocsr()
    n = X.shape[0]
    XT = X.T.tocsc()
    viz_sim = np.full(n, -1.0, dtype=np.float32)
    viz_id = np.full(n, -1, dtype=int)
    glob_sim = np.full(n, -1.0, dtype=np.float32)
    glob_id = np.full(n, -1, dtype=int)
    pares = []
    for ini in range(0, n, tam):
        fim = min(ini + tam, n)
        S = (X[ini:fim] @ XT).toarray()
        idx = np.arange(ini, fim)
        S[np.arange(fim - ini), idx] = -1.0                       # sem auto-similaridade
        gj = S.argmax(axis=1)
        glob_sim[ini:fim], glob_id[ini:fim] = S[np.arange(fim - ini), gj], gj
        valido = isdev[None, :] & (grupo[None, :] != grupo[idx][:, None])
        Sv = np.where(valido, S, -1.0)
        vj = Sv.argmax(axis=1)
        viz_sim[ini:fim], viz_id[ini:fim] = Sv[np.arange(fim - ini), vj], vj
        ii, jj = np.nonzero((S >= limiar_pares) & (np.arange(n)[None, :] > idx[:, None]))
        for a, b in zip(ii + ini, jj):
            pares.append((a, b, float(S[a - ini, b])))
    pares = pd.DataFrame(pares, columns=["id_a", "id_b", "sim"]).sort_values("sim", ascending=False)
    return viz_sim, viz_id, glob_sim, glob_id, pares


def resumo_limiares(pares, y, grupo, n, viz_sim, viz_id, isdev):
    linhas = []
    for lim in LIMIARES:
        p = pares[pares.sim >= lim]
        docs = pd.unique(np.concatenate([p.id_a.to_numpy(), p.id_b.to_numpy()])) if len(p) else np.array([], int)
        conf = int((y[p.id_a.to_numpy()] != y[p.id_b.to_numpy()]).sum()) if len(p) else 0
        cruza = int((grupo[p.id_a.to_numpy()] != grupo[p.id_b.to_numpy()]).sum()) if len(p) else 0
        dev_val = isdev & (viz_sim >= lim)
        te = (~isdev) & (viz_sim >= lim)
        concorda = float((y[viz_id[dev_val | te]] == y[dev_val | te]).mean()) if (dev_val | te).any() else np.nan
        linhas.append({"limiar_cosseno": lim, "pares": len(p), "docs_envolvidos": len(docs),
                       "pct_corpus": len(docs) / n, "pares_rotulo_conflitante": conf,
                       "pares_entre_folds_ou_teste": cruza,
                       "pct_validacao_com_quase_copia_no_treino": float(dev_val.sum() / isdev.sum()),
                       "pct_teste_com_quase_copia_no_dev": float(te.sum() / (~isdev).sum()),
                       "rotulo_igual_ao_da_copia": concorda})
    return pd.DataFrame(linhas)


# ------------------------------------------------------------------ 4. impacto
def _ajusta(tr, va, X, y):
    m = pipeline_lr().fit(X[tr], y[tr])
    return va, m.predict_proba(X[va])[:, 1]


def cv_agrupada(Xd, yd, pares, ids_dev, lim=0.80, n_jobs=2):
    pos = {g: i for i, g in enumerate(ids_dev)}
    p = pares[(pares.sim >= lim) & pares.id_a.isin(list(pos)) & pares.id_b.isin(list(pos))]
    a = p.id_a.map(pos).to_numpy()
    b = p.id_b.map(pos).to_numpy()
    n = len(ids_dev)
    g = coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
    ncomp, grupos = connected_components(g, directed=False)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    splits = list(sgkf.split(Xd, yd, grupos))
    res = Parallel(n_jobs=n_jobs)(delayed(_ajusta)(tr, va, Xd, yd) for tr, va in splits)
    oof = np.zeros(n)
    for va, pr in res:
        oof[va] = pr
    return oof, ncomp, int(pd.Series(grupos).value_counts().max())


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada", choices=ENTRADAS, default="noticia")
    ap.add_argument("--saida-base", default=None)
    args = ap.parse_args()
    saida = pasta_saida(args.saida_base)

    df = carregar_df()
    y = df["y"].to_numpy()
    n = len(df)
    ids_dev, ids_te, folds = dividir(y)
    grupo = np.full(n, 5)
    grupo[ids_dev] = folds
    isdev = np.zeros(n, bool)
    isdev[ids_dev] = True
    textos = montar_texto(df, args.entrada)
    md, js = [], {}

    print("1. duplicatas exatas...")
    ex = duplicatas_exatas(df, grupo)
    ex.to_csv(saida / "duplicacao_1_exatas.csv", index=False, encoding="utf-8")
    md += ["### 1. Duplicatas exatas residuais (base já deduplicada por URL e texto)", "", tabela_md(ex), ""]
    exi = ex.set_index("chave")
    conf = int(exi.loc[[c for c in exi.index if c.startswith(("Noticia", "Titulo+", "URL"))],
                       "grupos_com_rotulos_conflitantes"].sum())
    md += [veredito("CRITICO" if conf else "OK",
                    f"{conf} grupo(s) de TEXTO/URL idêntico com rótulos conflitantes." if conf else
                    "nenhum texto/URL idêntico com rótulos conflitantes (o corpus é consistente)."), ""]
    t_rep = exi.loc["Titulo (normalizado)"]
    if t_rep["grupos_com_rotulos_conflitantes"]:
        md += [veredito("INFO", f"{int(t_rep['grupos_com_rotulos_conflitantes'])} título(s) idêntico(s) aparecem com rótulos "
                        "diferentes (textos distintos sob o mesmo título genérico) — não é duplicação do conteúdo."), ""]
    md += [veredito("ATENCAO" if t_rep["docs_excedentes"] > 0 else "OK",
                    f"{int(t_rep['docs_excedentes'])} títulos repetidos (normalizados) permanecem na base — o protocolo "
                    "deduplicou por URL e texto, não por título; "
                    f"{int(t_rep['grupos_que_cruzam_folds_ou_teste'])} desses grupos cruzam folds/teste."), ""]

    print("2/3. quase-duplicatas (cosseno TF-IDF)...")
    t0 = time.time()
    viz_sim, viz_id, glob_sim, glob_id, pares = similaridades(textos, grupo, isdev, y, df)
    print(f"   {len(pares)} pares com cosseno >= {min(LIMIARES)} ({time.time() - t0:.0f}s)")
    resumo = resumo_limiares(pares, y, grupo, n, viz_sim, viz_id, isdev)
    resumo.to_csv(saida / "duplicacao_2_quase_duplicatas.csv", index=False, encoding="utf-8")
    top = pares.head(3000).copy()
    for lado in ("a", "b"):
        ids = top[f"id_{lado}"].to_numpy()
        top[f"y_{lado}"] = y[ids]
        top[f"dominio_{lado}"] = df["Dominio"].to_numpy()[ids]
        top[f"split_{lado}"] = np.where(grupo[ids] == 5, "teste", "fold" + pd.Series(grupo[ids]).astype(str))
        top[f"titulo_{lado}"] = df["Titulo"].fillna("").str.slice(0, 90).to_numpy()[ids]
    top.to_csv(saida / "duplicacao_pares_quase_duplicados.csv", index=False, encoding="utf-8")
    md += ["### 2. Quase-duplicatas e vazamento entre divisões (cosseno TF-IDF, entrada = " + args.entrada + ")", "",
           tabela_md(resumo), ""]
    r90 = resumo[resumo.limiar_cosseno == 0.90].iloc[0]
    nivel = "ATENCAO" if r90["pct_validacao_com_quase_copia_no_treino"] >= 0.03 else "OK"
    md += [veredito(nivel, f"{r90['pct_validacao_com_quase_copia_no_treino']:.1%} dos documentos de validação têm uma "
                    f"quase-cópia (cos ≥ 0,90) no treino; em {r90['rotulo_igual_ao_da_copia']:.1%} dos casos a cópia tem o "
                    "mesmo rótulo."), ""]
    js["quase_duplicatas"] = resumo.to_dict("records")

    print("4. impacto no desempenho (CV padrão x CV agrupada)...")
    Xd, yd = textos[ids_dev], y[ids_dev]
    oof = cv_oof(Xd, yd, folds, pipeline_lr)
    auc_padrao = roc_auc_score(yd, oof)
    sim_dev = viz_sim[ids_dev]
    linhas = []
    for lim in LIMIARES:
        com = sim_dev >= lim
        linha = {"subconjunto": f"validação COM quase-cópia no treino (cos ≥ {lim:.2f})", "n": int(com.sum())}
        linha["roc_auc"] = float(roc_auc_score(yd[com], oof[com])) if com.sum() > 30 and len(np.unique(yd[com])) == 2 else np.nan
        linha["accuracy"] = float(((oof[com] >= 0.5) == yd[com]).mean()) if com.any() else np.nan
        linhas.append(linha)
        sem = ~com
        linhas.append({"subconjunto": f"validação SEM quase-cópia no treino (cos < {lim:.2f})", "n": int(sem.sum()),
                       "roc_auc": float(roc_auc_score(yd[sem], oof[sem])),
                       "accuracy": float(((oof[sem] >= 0.5) == yd[sem]).mean())})
    linhas.append({"subconjunto": "todos (CV do protocolo)", "n": len(yd), "roc_auc": float(auc_padrao),
                   "accuracy": float(((oof >= 0.5) == yd).mean())})
    oof_g, ncomp, maior = cv_agrupada(Xd, yd, pares, ids_dev, lim=0.80)
    auc_g = float(roc_auc_score(yd, oof_g))
    linhas.append({"subconjunto": f"CV AGRUPADA por componentes de quase-duplicatas (cos ≥ 0,80; {ncomp} grupos, maior={maior})",
                   "n": len(yd), "roc_auc": auc_g, "accuracy": float(((oof_g >= 0.5) == yd).mean())})
    imp = pd.DataFrame(linhas)
    imp.to_csv(saida / "duplicacao_4_impacto.csv", index=False, encoding="utf-8")
    delta = auc_padrao - auc_g
    md += ["### 3. Impacto no desempenho (LR + TF-IDF)", "", tabela_md(imp), "",
           veredito("OK" if delta < 0.003 else "ATENCAO",
                    f"AUC do protocolo {auc_padrao:.4f} x AUC com CV agrupada {auc_g:.4f} (queda de {delta:+.4f}). "
                    + ("A quase-duplicação não infla a validação de forma relevante." if delta < 0.003 else
                       "A validação está inflada por quase-cópias entre folds: reporte também a CV agrupada.")), ""]
    js["auc_protocolo"], js["auc_cv_agrupada"] = float(auc_padrao), auc_g

    print("5. títulos repetidos entre treino e validação...")
    tnorm = df["Titulo"].fillna("").map(normalizar).to_numpy()
    rep_treino = np.zeros(n, bool)
    concorda = np.full(n, np.nan)
    for k in range(6):
        alvo = np.where(grupo == k)[0]
        treino = np.where(isdev & (grupo != k))[0]
        mapa = pd.Series(y[treino], index=tnorm[treino]).groupby(level=0).mean()
        for i in alvo:
            if tnorm[i] and tnorm[i] in mapa.index:
                rep_treino[i] = True
                concorda[i] = float(round(mapa[tnorm[i]]) == y[i])
    t = pd.DataFrame([{"conjunto": "validação (dev, 5 folds)", "n": int(isdev.sum()),
                       "com_titulo_igual_no_treino": int(rep_treino[isdev].sum()),
                       "pct": float(rep_treino[isdev].mean()),
                       "rotulo_igual": float(np.nanmean(concorda[isdev]))},
                      {"conjunto": "teste", "n": int((~isdev).sum()),
                       "com_titulo_igual_no_treino": int(rep_treino[~isdev].sum()),
                       "pct": float(rep_treino[~isdev].mean()),
                       "rotulo_igual": float(np.nanmean(concorda[~isdev]))}])
    t.to_csv(saida / "duplicacao_5_titulos.csv", index=False, encoding="utf-8")
    md += ["### 4. Títulos repetidos entre treino e validação/teste", "", tabela_md(t), ""]

    salvar_md(saida, "duplicacao", "Auditoria de duplicação e quase-duplicação", md)
    salvar_json(saida, "duplicacao", js)
    print(f"\nRelatório: {saida / 'duplicacao.md'}")


if __name__ == "__main__":
    main()
