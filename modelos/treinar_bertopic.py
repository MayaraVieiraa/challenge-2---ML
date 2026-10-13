# -*- coding: utf-8 -*-
"""
treinar_bertopic.py — BERTopic como modelo base (Mayara)
=======================================================
O BERTopic é um modelo NÃO supervisionado de tópicos (embeddings -> UMAP -> HDBSCAN ->
c-TF-IDF). Aqui ele entra no projeto em dois papéis:

  1. MODELO BASE para o stacking: o BERTopic é ajustado (sem usar o rótulo) no treino de
     cada fold; cada notícia vira um vetor de "pertencimento aos tópicos" (+ massa de
     outlier); uma Regressão Logística treinada só no treino do fold estima P(Fake).
     É um modelo propositalmente DIFERENTE dos TF-IDF: enxerga tema/assunto, não palavras.
  2. INSTRUMENTO DE AUDITORIA: tabela tópico x rótulo (modelo final, desenvolvimento)
     mostra quais assuntos são quase 100% Fake ou Real — dependência temática/editorial.

Protocolo: base NoRemovalWords_tratada_dedup | texto = Titulo + Noticia (ou outra --entrada) |
mesmo split dos outros modelos (teste 80/20 reservado + StratifiedKFold(5, shuffle, 42) no
desenvolvimento) | classe positiva = Fake | mesmas métricas (metricas.json) |
calibração nativa da LR (sem CalibratedClassifierCV), como no baseline.

Embeddings (não usam rótulo; calculados uma vez para todos os documentos e guardados em cache):
  --backend sbert       (padrão) sentence-transformers multilíngue; baixa o modelo do Hugging
                        Face na 1ª execução (precisa de internet): pip install sentence-transformers
  --backend tfidf-svd   alternativa 100% offline (TF-IDF + SVD de 100 dim). Útil para testar o
                        pipeline sem baixar modelos; o SVD é ajustado em todos os textos, SEM
                        rótulos (vazamento de rótulo = nenhum; é transdutivo apenas no texto).

Saídas em modelos_outputs/bertopic/:
    metricas.json, reliability.png, matriz_confusao.png, classification_report.txt,
    oof_bertopic.csv, teste_bertopic.csv,
    topicos_info.csv (tópico, tamanho, %Fake, palavras), topicos_vs_rotulo.png,
    embeddings_<backend>_<hash>.npy (cache)

Uso:
    python modelos/treinar_bertopic.py                       # sbert (precisa de internet)
    python modelos/treinar_bertopic.py --backend tfidf-svd   # offline
"""
import os
import argparse
import hashlib
import json
import sys
import time
import unicodedata
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             average_precision_score, brier_score_loss,
                             classification_report, confusion_matrix, log_loss,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _divisao import dividir_grupos, grupos_de  # noqa: E402
from _entrada import ENTRADAS, montar_texto  # noqa: E402

SEED = 42
RAIZ = Path(__file__).resolve().parents[1]
DADOS = RAIZ / "dados_tratados" / os.environ.get("BASE_DADOS", "FakeRecogna2_extrativa_casada.csv")
NOME = "bertopic"
SAIDA = RAIZ / "modelos_outputs" / NOME
CLASSES = ["Real", "Fake"]
N_BOOT = 1000
MODELO_EMBEDDING = "paraphrase-multilingual-MiniLM-L12-v2"

# stopwords mínimas do português, só para os rótulos dos tópicos (c-TF-IDF) ficarem legíveis
STOP_PT = ["a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das", "em", "no", "na",
           "nos", "nas", "por", "para", "com", "sem", "sob", "sobre", "e", "ou", "mas", "que", "se", "como",
           "ao", "aos", "à", "às", "ser", "estar", "ter", "haver", "ir", "foi", "é", "são", "há", "já", "não",
           "mais", "muito", "também", "só", "seu", "sua", "seus", "suas", "ele", "ela", "eles", "elas", "isso",
           "este", "esta", "esse", "essa", "n", "ano", "dia", "vez"]


# ------------------------------------------------------------------ dados / split
def carregar(entrada):
    df = pd.read_csv(DADOS)
    texto = montar_texto(df, entrada)
    y = (df["Rotulo"].astype(str).str.strip().str.lower() == "fake").astype(int).to_numpy()
    return texto, y


def dividir(y):
    return dividir_grupos(y, grupos_de(DADOS, len(y)))


# ------------------------------------------------------------------ embeddings
def obter_embeddings(X, args):
    """Embeddings de todos os documentos (sem rótulos), com cache em disco."""
    chave = hashlib.md5((args.backend + args.modelo_embedding + args.entrada +
                         str(len(X)) + X[0][:50] + X[-1][:50]).encode("utf-8")).hexdigest()[:10]
    cache = SAIDA / f"embeddings_{args.backend}_{chave}.npy"
    if cache.exists():
        print(f"Embeddings em cache: {cache.name}")
        return np.load(cache)
    t0 = time.time()
    if args.backend == "sbert":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise SystemExit("Instale: pip install sentence-transformers  "
                             "(ou use --backend tfidf-svd para rodar offline)") from e
        print(f"Calculando embeddings com '{args.modelo_embedding}' (pode levar alguns minutos)...")
        enc = SentenceTransformer(args.modelo_embedding)
        emb = enc.encode(list(X), batch_size=args.batch_size, show_progress_bar=True,
                         normalize_embeddings=True)
    else:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import Normalizer
        print("Calculando embeddings TF-IDF + SVD(100) (offline)...")
        tf = TfidfVectorizer(min_df=5, max_df=0.9, sublinear_tf=True, strip_accents="unicode")
        emb = Normalizer().fit_transform(TruncatedSVD(100, random_state=SEED).fit_transform(tf.fit_transform(X)))
    emb = np.asarray(emb, dtype=np.float32)
    SAIDA.mkdir(parents=True, exist_ok=True)
    np.save(cache, emb)
    print(f"Embeddings {emb.shape} prontos em {time.time() - t0:.0f}s")
    return emb


# ------------------------------------------------------------------ BERTopic
def sem_acento(docs):
    """O BERTopic remove caracteres não-ASCII ao montar os rótulos dos tópicos ("ministério" -> "ministrio");
    tirar os acentos antes preserva as palavras. Os embeddings já foram calculados com o texto original."""
    return [unicodedata.normalize("NFKD", d).encode("ascii", "ignore").decode("ascii") for d in docs]


def novo_bertopic(args):
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP
    return BERTopic(
        embedding_model=None,  # embeddings pré-calculados são passados em fit/transform
        umap_model=UMAP(n_neighbors=15, n_components=args.n_umap, min_dist=0.0, metric="cosine",
                        random_state=SEED),
        hdbscan_model=HDBSCAN(min_cluster_size=args.min_cluster_size, metric="euclidean",
                              cluster_selection_method="eom", prediction_data=True),
        # O BERTopic aplica este vetorizador a UM documento por TOPICO (nao por noticia). Com min_df=5 e
        # max_df=0.9 ele exige 6+ topicos e quebra quando um fold gera poucos ("max_df corresponds to
        # < documents than min_df"). min_df=1/max_df=1.0 so afeta as palavras que rotulam os topicos,
        # nao a atribuicao de topico nem as features do classificador.
        vectorizer_model=CountVectorizer(ngram_range=(1, 2), min_df=1, max_df=1.0,
                                         strip_accents="unicode", stop_words=STOP_PT),
        calculate_probabilities=True,
        nr_topics=args.nr_topicos,
        top_n_words=10,
        verbose=False,
    )


def features_topicos(modelo, docs, emb, com_embeddings):
    """Vetor por documento: pertencimento a cada tópico + massa de outlier (+ embedding opcional)."""
    topicos, probs = modelo.transform(sem_acento(docs), embeddings=emb)
    topicos = np.asarray(topicos)
    n_top = len([t for t in modelo.get_topics() if t != -1])
    if probs is None or np.ndim(probs) != 2:       # fallback: one-hot do tópico atribuído
        probs = np.zeros((len(docs), n_top))
        for i, t in enumerate(topicos):
            if t >= 0:
                probs[i, t] = 1.0
    probs = np.asarray(probs, dtype=np.float64)
    outlier = np.clip(1.0 - probs.sum(axis=1, keepdims=True), 0, 1)
    f = np.hstack([probs, outlier, (topicos == -1).astype(float).reshape(-1, 1)])
    if com_embeddings:
        f = np.hstack([f, emb])
    return f, topicos


def classificador():
    return Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(C=1.0, max_iter=3000, random_state=SEED))])


# ------------------------------------------------------------------ métricas
def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def metricas_fold(y, p):
    pred = (p >= 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    m = {"accuracy": accuracy_score(y, pred), "roc_auc": roc_auc_score(y, p),
         "pr_auc": average_precision_score(y, p), "brier": brier_score_loss(y, p),
         "log_loss": log_loss(y, np.clip(p, 1e-15, 1 - 1e-15)), "ece": ece(y, p)}
    for i, nome in enumerate(CLASSES):
        m[f"precision_{nome}"], m[f"recall_{nome}"], m[f"f1_{nome}"] = prec[i], rec[i], f1[i]
    return m


def ic95_auc(y, p, n=N_BOOT):
    rng = np.random.default_rng(SEED)
    aucs = []
    while len(aucs) < n:
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            aucs.append(roc_auc_score(y[idx], p[idx]))
    return [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]


# ------------------------------------------------------------------ gráficos / tabelas
def salvar_reliability(y, p):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1.2, 1]})
    axs[0].plot([0, 1], [0, 1], "--", c="gray", label="Calibração perfeita")
    real, prevista = calibration_curve(y, p, n_bins=10)
    axs[0].plot(prevista, real, "o-", c="#7b3fa0",
                label=f"BERTopic+LR (Brier {brier_score_loss(y, p):.4f}, ECE {ece(y, p):.4f})")
    axs[0].set(xlabel="Probabilidade prevista de Fake", ylabel="Fração observada de Fake",
               title="BERTopic: reliability diagram (OOF, 5 folds)")
    axs[0].legend(loc="upper left")
    axs[1].hist(p[y == 0], bins=30, alpha=.6, label="Real", color="#2a6fdb")
    axs[1].hist(p[y == 1], bins=30, alpha=.6, label="Fake", color="#c0392b")
    axs[1].set(xlabel="P(Fake) fora-do-fold", ylabel="Notícias", title="Distribuição das probabilidades")
    axs[1].legend()
    fig.tight_layout()
    fig.savefig(SAIDA / "reliability.png", dpi=150)
    plt.close(fig)


def salvar_matriz(y, p):
    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    ConfusionMatrixDisplay(confusion_matrix(y, (p >= 0.5).astype(int), labels=[0, 1]),
                           display_labels=CLASSES).plot(ax=ax, colorbar=False)
    ax.set_title("BERTopic+LR — matriz de confusão (OOF)")
    fig.tight_layout()
    fig.savefig(SAIDA / "matriz_confusao.png", dpi=150)
    plt.close(fig)


def salvar_topicos(modelo, topicos_dev, y_dev):
    """Tabela tópico x rótulo (modelo final ajustado em todo o desenvolvimento)."""
    info = modelo.get_topic_info()
    linhas = []
    for _, r in info.iterrows():
        t = int(r["Topic"])
        m = topicos_dev == t
        if not m.any():
            continue
        palavras = [w for w, _ in (modelo.get_topic(t) or [])][:10]
        pf = float(y_dev[m].mean())
        linhas.append({"topico": t, "n_docs": int(m.sum()), "pct_fake": pf,
                       "pureza": max(pf, 1 - pf), "palavras_top": ", ".join(palavras)})
    tab = pd.DataFrame(linhas).sort_values("n_docs", ascending=False)
    tab.to_csv(SAIDA / "topicos_info.csv", index=False, encoding="utf-8")

    t = tab[tab.topico != -1].head(25).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(t) + 1.5))
    ax.barh([f"T{r.topico}: {r.palavras_top[:42]}" for r in t.itertuples()], t.pct_fake * 100,
            color=["#c0392b" if v >= .5 else "#2a6fdb" for v in t.pct_fake])
    ax.axvline(50, c="gray", ls="--", lw=1)
    ax.set(xlabel="% de notícias Fake no tópico (desenvolvimento)",
           title="BERTopic: pureza de rótulo dos 25 maiores tópicos")
    fig.tight_layout()
    fig.savefig(SAIDA / "topicos_vs_rotulo.png", dpi=140)
    plt.close(fig)
    return tab


# ------------------------------------------------------------------ main
def main():
    global SAIDA
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["sbert", "tfidf-svd"], default="sbert")
    ap.add_argument("--modelo-embedding", default=MODELO_EMBEDDING)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--min-cluster-size", type=int, default=10)
    ap.add_argument("--n-umap", type=int, default=5, help="dimensões do UMAP")
    ap.add_argument("--nr-topicos", type=int, default=None, help="reduz para N tópicos (padrão: automático)")
    ap.add_argument("--com-embeddings", action="store_true",
                    help="concatena o embedding denso às probabilidades de tópico (modelo mais forte, menos 'puro' BERTopic)")
    ap.add_argument("--entrada", choices=ENTRADAS, default="noticia")
    ap.add_argument("--saida-base", default=None)
    ap.add_argument("--sem-teste", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    SAIDA = Path(args.saida_base or RAIZ / "modelos_outputs") / NOME
    SAIDA.mkdir(parents=True, exist_ok=True)

    X, y = carregar(args.entrada)
    ids_dev, ids_te, folds = dividir(y)
    X_dev, y_dev = X[ids_dev], y[ids_dev]
    print(f"Documentos: {len(y)} | Desenvolvimento: {len(ids_dev)} | Teste (reservado): {len(ids_te)}")
    emb = obter_embeddings(X, args)
    E_dev, E_te = emb[ids_dev], emb[ids_te]

    oof = np.zeros(len(y_dev))
    por_fold, info_folds = [], []
    t0 = time.time()
    for k in range(5):
        tr, va = folds != k, folds == k
        bt = novo_bertopic(args)
        bt.fit_transform(sem_acento(X_dev[tr]), embeddings=E_dev[tr])           # sem rótulos
        F_tr, top_tr = features_topicos(bt, X_dev[tr], E_dev[tr], args.com_embeddings)
        F_va, _ = features_topicos(bt, X_dev[va], E_dev[va], args.com_embeddings)
        clf = classificador().fit(F_tr, y_dev[tr])
        oof[va] = clf.predict_proba(F_va)[:, 1]
        por_fold.append(metricas_fold(y_dev[va], oof[va]))
        n_top = len([t for t in bt.get_topics() if t != -1])
        info_folds.append({"fold": k, "n_topicos": n_top, "pct_outliers_treino": float((top_tr == -1).mean())})
        print(f"fold {k + 1}/5  tópicos={n_top}  outliers={info_folds[-1]['pct_outliers_treino']:.1%}  "
              f"AUC={por_fold[-1]['roc_auc']:.4f}  Brier={por_fold[-1]['brier']:.4f}  ({time.time() - t0:.0f}s)")

    tabela = pd.DataFrame(por_fold)
    resultado = {
        "modelo": "BERTopic (topic-probabilities) + LogisticRegression",
        "classe_positiva": "Fake",
        "entrada": args.entrada,
        "escolhido": "bertopic_lr",
        "n_amostras_validacao_cruzada": int(len(y_dev)),
        "validacao": "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42, grupos = URL) sobre os 80% de desenvolvimento; "
                     "o BERTopic é reajustado (sem rótulos) no treino de cada fold",
        "hiperparametros": {"backend": args.backend, "modelo_embedding": args.modelo_embedding,
                            "min_cluster_size": args.min_cluster_size, "n_umap": args.n_umap,
                            "nr_topicos": args.nr_topicos, "com_embeddings": args.com_embeddings},
        "calibracao": "nativa (LR sem CalibratedClassifierCV)",
        "topicos_por_fold": info_folds,
        "bertopic_lr": {
            "media": tabela.mean().to_dict(),
            "desvio_padrao": tabela.std(ddof=1).to_dict(),
            "roc_auc_ic95": ic95_auc(y_dev, oof),
            "pct_previsoes_extremas": float(np.mean((oof < 0.01) | (oof > 0.99))),
        },
    }
    (SAIDA / "metricas.json").write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    (SAIDA / "classification_report.txt").write_text(
        "===== BERTopic + LR (previsões fora do fold, limiar 0,5) =====\n" +
        classification_report(y_dev, (oof >= 0.5).astype(int), target_names=CLASSES, digits=4), encoding="utf-8")
    salvar_reliability(y_dev, oof)
    salvar_matriz(y_dev, oof)

    col = f"p_fake_{NOME}"
    pd.DataFrame({"id": ids_dev, "fold": folds, "y_fake": y_dev, col: oof}).to_csv(
        SAIDA / f"oof_{NOME}.csv", index=False)

    # modelo final no desenvolvimento inteiro: tabela de tópicos + previsões do teste
    bt = novo_bertopic(args)
    top_dev = np.asarray(bt.fit_transform(sem_acento(X_dev), embeddings=E_dev)[0])
    tab = salvar_topicos(bt, top_dev, y_dev)
    if not args.sem_teste:
        F_dev, _ = features_topicos(bt, X_dev, E_dev, args.com_embeddings)
        F_te, _ = features_topicos(bt, X[ids_te], E_te, args.com_embeddings)
        clf = classificador().fit(F_dev, y_dev)
        pd.DataFrame({"id": ids_te, "y_fake": y[ids_te], col: clf.predict_proba(F_te)[:, 1]}).to_csv(
            SAIDA / f"teste_{NOME}.csv", index=False)

    m, s = resultado["bertopic_lr"]["media"], resultado["bertopic_lr"]["desvio_padrao"]
    lo, hi = resultado["bertopic_lr"]["roc_auc_ic95"]
    print("\n=== BERTopic + LR (média ± dp entre folds) ===")
    for c in ("roc_auc", "pr_auc", "brier", "ece", "accuracy"):
        print(f"  {c:9s} {m[c]:.4f} ± {s[c]:.4f}")
    print(f"  ROC-AUC IC95% (bootstrap {N_BOOT}x): [{lo:.4f}, {hi:.4f}]")
    puros = tab[(tab.topico != -1) & (tab.n_docs >= 30) & (tab.pureza >= 0.95)]
    print(f"\nTópicos (modelo final): {int((tab.topico != -1).sum())} | com >=30 docs e >=95% de um só rótulo: {len(puros)} "
          f"({puros.n_docs.sum()} docs)")
    print(f"Saídas em {SAIDA}")


if __name__ == "__main__":
    main()