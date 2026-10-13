# -*- coding: utf-8 -*-
"""
treinar_baseline.py — Baseline: TF-IDF + Regressão Logística (Mayara)
=====================================================================
Segue o protocolo comum (modelos/README.md):

  * base ......... dados_tratados/NoRemovalWords_tratada_dedup.csv
  * entrada ...... Titulo + Noticia (concatenados). Nenhuma outra coluna é usada.
  * vetorização .. TfidfVectorizer(ngram_range=(1,2), min_df=5, max_df=0.9,
                   sublinear_tf=True, strip_accents="unicode") — ajustado DENTRO de
                   cada fold (Pipeline), nunca olhando a validação.
  * split ........ teste final 80/20 estratificado (reservado) + StratifiedKFold(5,
                   shuffle=True, random_state=42) nos 80% de desenvolvimento.
  * calibração ... NATIVA: a Regressão Logística é usada SEM CalibratedClassifierCV;
                   medimos o quão calibradas já saem as probabilidades.
  * alvo ......... y = 1 se Rotulo == "Fake" (classe positiva = Fake).

Saídas em modelos_outputs/baseline/:
    metricas.json, reliability.png, matriz_confusao.png,
    classification_report.txt, coeficientes.csv,
    oof_baseline.csv   (probabilidades fora-do-fold, para comparação/stacking)
    teste_baseline.csv (probabilidades no teste; NENHUMA métrica de teste é
                        calculada aqui — o teste só é avaliado na entrega final)

Uso (a partir de qualquer pasta):
    python modelos/treinar_baseline.py
"""
import os
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             average_precision_score, brier_score_loss,
                             classification_report, confusion_matrix, log_loss,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _divisao import dividir_grupos, grupos_de  # noqa: E402
from _entrada import ENTRADAS, montar_texto  # noqa: E402

SEED = 42
RAIZ = Path(__file__).resolve().parents[1]
DADOS = RAIZ / "dados_tratados" / os.environ.get("BASE_DADOS", "FakeRecogna2_extrativa_casada.csv")
NOME = "baseline"
SAIDA = RAIZ / "modelos_outputs" / NOME
CLASSES = ["Real", "Fake"]  # 0 = Real, 1 = Fake (classe positiva)
N_BOOT = 1000


# ------------------------------------------------------------------ dados / split
def carregar(entrada="noticia"):
    df = pd.read_csv(DADOS)
    texto = pd.Series(montar_texto(df, entrada))
    y = (df["Rotulo"].astype(str).str.strip().str.lower() == "fake").astype(int).to_numpy()
    return texto.to_numpy(), y


def dividir(y):
    """Mesmo split dos outros modelos: ids = posição da linha no CSV."""
    return dividir_grupos(y, grupos_de(DADOS, len(y)))


# ------------------------------------------------------------------ modelo
def vetorizador():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.9,
                           sublinear_tf=True, strip_accents="unicode")


def montar_modelo():
    # C=1.0 (padrão): é um baseline, sem ajuste de hiperparâmetros.
    return Pipeline([("tfidf", vetorizador()),
                     ("lr", LogisticRegression(C=1.0, max_iter=2000, random_state=SEED))])


# ------------------------------------------------------------------ métricas
def ece(y, p, bins=10):
    """Expected Calibration Error (bins uniformes)."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def metricas_fold(y, p):
    pred = (p >= 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    m = {
        "accuracy": accuracy_score(y, pred),          # só referência
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, np.clip(p, 1e-15, 1 - 1e-15)),
        "ece": ece(y, p),
    }
    for i, nome in enumerate(CLASSES):
        m[f"precision_{nome}"] = prec[i]
        m[f"recall_{nome}"] = rec[i]
        m[f"f1_{nome}"] = f1[i]
    return m


def ic95_auc(y, p, n=N_BOOT):
    rng = np.random.default_rng(SEED)
    aucs = []
    while len(aucs) < n:
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            aucs.append(roc_auc_score(y[idx], p[idx]))
    return [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]


# ------------------------------------------------------------------ gráficos
def salvar_reliability(y, p, titulo):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axs[0]
    ax.plot([0, 1], [0, 1], "--", c="gray", label="Calibração perfeita")
    real, prevista = calibration_curve(y, p, n_bins=10)
    ax.plot(prevista, real, "o-", c="#2a6fdb",
            label=f"LR nativa (Brier {brier_score_loss(y, p):.4f}, ECE {ece(y, p):.4f})")
    ax.set(xlabel="Probabilidade prevista de Fake", ylabel="Fração observada de Fake",
           title=titulo)
    ax.legend(loc="upper left")
    axs[1].hist(p[y == 0], bins=30, alpha=.6, label="Real", color="#2a6fdb")
    axs[1].hist(p[y == 1], bins=30, alpha=.6, label="Fake", color="#c0392b")
    axs[1].set(xlabel="P(Fake) fora-do-fold", ylabel="Notícias", title="Distribuição das probabilidades")
    axs[1].legend()
    fig.tight_layout()
    fig.savefig(SAIDA / "reliability.png", dpi=150)
    plt.close(fig)


def salvar_matriz(y, p):
    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    cm = confusion_matrix(y, (p >= 0.5).astype(int), labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, colorbar=False)
    ax.set_title("Baseline LR — matriz de confusão (OOF, limiar 0,5)")
    fig.tight_layout()
    fig.savefig(SAIDA / "matriz_confusao.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ coeficientes
def salvar_coeficientes(modelo, X_dev, y_dev):
    """Coeficientes do modelo final (ajustado em todo o desenvolvimento).
    Coef > 0 empurra para Fake; < 0 empurra para Real."""
    tfidf, lr = modelo.named_steps["tfidf"], modelo.named_steps["lr"]
    termos = tfidf.get_feature_names_out()
    coef = lr.coef_.ravel()
    # frequência de documentos por classe (presença do termo)
    Xb = tfidf.transform(X_dev)
    Xb.data[:] = 1
    df_fake = np.asarray(Xb[y_dev == 1].sum(axis=0)).ravel()
    df_real = np.asarray(Xb[y_dev == 0].sum(axis=0)).ravel()
    tab = pd.DataFrame({
        "termo": termos,
        "coef": coef,
        "odds_ratio": np.exp(coef),
        "aponta_para": np.where(coef > 0, "Fake", "Real"),
        "docs_fake": df_fake.astype(int),
        "docs_real": df_real.astype(int),
        "pct_docs_fake": df_fake / max(int((y_dev == 1).sum()), 1),
        "pct_docs_real": df_real / max(int((y_dev == 0).sum()), 1),
    }).sort_values("coef", ascending=False).reset_index(drop=True)
    tab.to_csv(SAIDA / "coeficientes.csv", index=False, encoding="utf-8")
    return tab


# ------------------------------------------------------------------ main
def main():
    global SAIDA
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sem-teste", action="store_true",
                    help="não gera as probabilidades do conjunto de teste")
    ap.add_argument("--entrada", choices=ENTRADAS, default="noticia",
                    help="texto de entrada (padrão: só Noticia, sem o vazamento do título; 'titulo_noticia' = protocolo original)")
    ap.add_argument("--saida-base", default=None,
                    help="pasta-base das saídas (padrão: <raiz>/modelos_outputs)")
    args = ap.parse_args()
    SAIDA = Path(args.saida_base or RAIZ / "modelos_outputs") / NOME

    SAIDA.mkdir(parents=True, exist_ok=True)
    X, y = carregar(args.entrada)
    ids_dev, ids_te, folds = dividir(y)
    X_dev, y_dev = X[ids_dev], y[ids_dev]
    print(f"Documentos: {len(y)} | Fake: {y.sum()} ({y.mean():.1%}) | "
          f"Desenvolvimento: {len(ids_dev)} | Teste (reservado): {len(ids_te)}")

    oof = np.zeros(len(y_dev))
    por_fold = []
    t0 = time.time()
    for k in range(5):
        tr, va = folds != k, folds == k
        modelo = montar_modelo().fit(X_dev[tr], y_dev[tr])
        oof[va] = modelo.predict_proba(X_dev[va])[:, 1]
        por_fold.append(metricas_fold(y_dev[va], oof[va]))
        print(f"fold {k + 1}/5  AUC={por_fold[-1]['roc_auc']:.4f}  "
              f"Brier={por_fold[-1]['brier']:.4f}  ({time.time() - t0:.0f}s)")

    tabela = pd.DataFrame(por_fold)
    resultado = {
        "modelo": "LogisticRegression + TF-IDF (baseline)",
        "classe_positiva": "Fake",
        "entrada": args.entrada,
        "escolhido": "lr_nativa",
        "n_amostras_validacao_cruzada": int(len(y_dev)),
        "validacao": "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42, grupos = URL) sobre os 80% de desenvolvimento",
        "hiperparametros": {"C": 1.0, "max_iter": 2000,
                            "tfidf": "ngram(1,2), min_df=5, max_df=0.9, sublinear_tf, strip_accents=unicode"},
        "calibracao": "nativa (sem CalibratedClassifierCV)",
        "lr_nativa": {
            "media": tabela.mean().to_dict(),
            "desvio_padrao": tabela.std(ddof=1).to_dict(),
            "roc_auc_ic95": ic95_auc(y_dev, oof),
            "metricas_oof_agregadas": metricas_fold(y_dev, oof),
            "pct_previsoes_extremas": float(np.mean((oof < 0.01) | (oof > 0.99))),
        },
    }
    (SAIDA / "metricas.json").write_text(json.dumps(resultado, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    (SAIDA / "classification_report.txt").write_text(
        "===== Baseline LR (previsões fora do fold, limiar 0,5) =====\n" +
        classification_report(y_dev, (oof >= 0.5).astype(int), target_names=CLASSES, digits=4),
        encoding="utf-8")
    salvar_reliability(y_dev, oof, "Baseline LR: reliability diagram (OOF, 5 folds)")
    salvar_matriz(y_dev, oof)

    # saídas para comparação / stacking
    col = f"p_fake_{NOME}"
    pd.DataFrame({"id": ids_dev, "fold": folds, "y_fake": y_dev, col: oof}).to_csv(
        SAIDA / f"oof_{NOME}.csv", index=False)

    # modelo final (todo o desenvolvimento): coeficientes + previsões do teste
    final = montar_modelo().fit(X_dev, y_dev)
    coef = salvar_coeficientes(final, X_dev, y_dev)
    if not args.sem_teste:
        pd.DataFrame({"id": ids_te, "y_fake": y[ids_te], col: final.predict_proba(X[ids_te])[:, 1]}).to_csv(
            SAIDA / f"teste_{NOME}.csv", index=False)

    m, s = resultado["lr_nativa"]["media"], resultado["lr_nativa"]["desvio_padrao"]
    print("\n=== Baseline LR (média ± dp entre folds) ===")
    for c in ("roc_auc", "pr_auc", "brier", "ece", "accuracy"):
        print(f"  {c:9s} {m[c]:.4f} ± {s[c]:.4f}")
    lo, hi = resultado["lr_nativa"]["roc_auc_ic95"]
    print(f"  ROC-AUC IC95% (bootstrap {N_BOOT}x): [{lo:.4f}, {hi:.4f}]")
    print("\nTermos que mais empurram para FAKE:   " + ", ".join(coef.head(15)["termo"]))
    print("Termos que mais empurram para REAL:   " + ", ".join(coef.tail(15)["termo"][::-1]))
    print(f"\nSaídas em {SAIDA}")


if __name__ == "__main__":
    main()