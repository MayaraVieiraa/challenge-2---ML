# -*- coding: utf-8 -*-
"""
treinar_svm.py — Linear SVM calibrado: TF-IDF + LinearSVC + CalibratedClassifierCV (Rayssa)
===========================================================================================
Integra o script original da Rayssa (guardado em modelos/originais/) ao PROTOCOLO COMUM
(modelos/README.md), para que os resultados sejam comparáveis com os outros modelos.

O que foi mantido do script original
    * LinearSVC + CalibratedClassifierCV (cv=5 interno, ensemble=True)
    * grade de C = [0.1, 0.3, 1, 3, 10] e os dois métodos, "sigmoid" e "isotonic",
      ambos reportados (exigência do protocolo)
    * escolha da configuração final pelo MENOR Brier OOF, com a regra de desempate
      pró-sigmoid (diferença < 0,0005 em Brier => sigmoid)
    * alvo y = 1 se Rotulo == "Fake"

O que foi alinhado ao protocolo (diferenças em relação ao script original)
    * base: NoRemovalWords_tratada_dedup.csv (o original apontava para FakeRecogna_...)
    * entrada: Titulo + Noticia (o original usava só Noticia)
    * TF-IDF: ngram(1,2), min_df=5, max_df=0.9, sublinear_tf, strip_accents="unicode"
      (o original: min_df=2, max_df=0.95, max_features=200k, sem strip_accents)
    * split: teste 80/20 + StratifiedKFold(5, shuffle, 42) calculados em memória, de
      forma idêntica aos outros scripts (não depende de split_stacking.csv)
    * métricas: as do protocolo (accuracy, roc_auc, pr_auc, brier, precision/recall/f1
      por classe, IC95% por bootstrap) em metricas.json, média ± dp entre folds

Eficiência: para cada fold o TF-IDF é ajustado UMA vez no treino do fold e reaproveitado
pelas 10 configurações (equivalente a refazer o Pipeline a cada configuração).

Saídas em modelos_outputs/svm/:
    metricas.json, reliability.png, matriz_confusao.png, classification_report.txt,
    comparacao_oof_svm.csv, oof_svm.csv, teste_svm.csv
Uso:
    python modelos/treinar_svm.py
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
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             average_precision_score, brier_score_loss,
                             classification_report, confusion_matrix, log_loss,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _divisao import dividir_grupos, grupos_de  # noqa: E402
from _entrada import ENTRADAS, montar_texto  # noqa: E402

SEED = 42
RAIZ = Path(__file__).resolve().parents[1]
DADOS = RAIZ / "dados_tratados" / os.environ.get("BASE_DADOS", "FakeRecogna2_extrativa_casada.csv")
NOME = "svm"
SAIDA = RAIZ / "modelos_outputs" / NOME
CLASSES = ["Real", "Fake"]
N_BOOT = 1000
GRADE_C = [0.1, 0.3, 1.0, 3.0, 10.0]
METODOS = ["sigmoid", "isotonic"]
TOLERANCIA_BRIER = 0.0005   # desempate pró-sigmoid (regra original da Rayssa)


# ------------------------------------------------------------------ dados / split
def carregar(entrada="noticia"):
    df = pd.read_csv(DADOS)
    texto = pd.Series(montar_texto(df, entrada))
    y = (df["Rotulo"].astype(str).str.strip().str.lower() == "fake").astype(int).to_numpy()
    return texto.to_numpy(), y


def dividir(y):
    return dividir_grupos(y, grupos_de(DADOS, len(y)))


# ------------------------------------------------------------------ modelo
VETORIZACAO = "protocolo"


def vetorizador():
    if VETORIZACAO == "rayssa":   # configuração EXATA do script original da Rayssa
        return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True,
                               max_features=200_000, strip_accents=None)
    return TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.9,
                           sublinear_tf=True, strip_accents="unicode")


def svm_calibrado(C, metodo):
    svm = LinearSVC(C=C, loss="squared_hinge", dual="auto", max_iter=20_000, random_state=SEED)
    return CalibratedClassifierCV(estimator=svm, method=metodo,
                                  cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                  ensemble=True)


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


# ------------------------------------------------------------------ gráficos
def salvar_reliability(y, oof_melhores, escolhido):
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.8))
    ax = axs[0]
    ax.plot([0, 1], [0, 1], "--", c="gray", label="Calibração perfeita")
    for (metodo, C, p), cor in zip(oof_melhores, ["#2a6fdb", "#e07a1f"]):
        real, prevista = calibration_curve(y, p, n_bins=10)
        ax.plot(prevista, real, "o-", c=cor,
                label=f"{metodo} (C={C}) — Brier {brier_score_loss(y, p):.4f}")
    ax.set(xlabel="Probabilidade prevista de Fake", ylabel="Fração observada de Fake",
           title="SVM calibrado: reliability diagram (OOF, 5 folds)")
    ax.legend(loc="upper left", fontsize=8)
    metodo, C, p = [t for t in oof_melhores if t[0] == escolhido][0]
    axs[1].hist(p[y == 0], bins=30, alpha=.6, label="Real", color="#2a6fdb")
    axs[1].hist(p[y == 1], bins=30, alpha=.6, label="Fake", color="#c0392b")
    axs[1].set(xlabel="P(Fake) fora-do-fold", ylabel="Notícias",
               title=f"Distribuição ({metodo}, C={C})")
    axs[1].legend()
    fig.tight_layout()
    fig.savefig(SAIDA / "reliability.png", dpi=150)
    plt.close(fig)


def salvar_matriz(y, oof_melhores):
    fig, axs = plt.subplots(1, len(oof_melhores), figsize=(5 * len(oof_melhores), 4.5))
    for ax, (metodo, C, p) in zip(np.atleast_1d(axs), oof_melhores):
        cm = confusion_matrix(y, (p >= 0.5).astype(int), labels=[0, 1])
        ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, colorbar=False)
        ax.set_title(f"{metodo} (C={C})")
    fig.tight_layout()
    fig.savefig(SAIDA / "matriz_confusao.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    global SAIDA, VETORIZACAO
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sem-teste", action="store_true")
    ap.add_argument("--vetorizacao", choices=["protocolo", "rayssa"], default="protocolo",
                    help="'rayssa' = TF-IDF idêntico ao script original dela (min_df=2, max_df=0.95, 200k features, com acentos)")
    ap.add_argument("--entrada", choices=ENTRADAS, default="noticia",
                    help="texto de entrada (padrão: só Noticia, sem o vazamento do título; 'titulo_noticia' = protocolo original)")
    ap.add_argument("--saida-base", default=None,
                    help="pasta-base das saídas (padrão: <raiz>/modelos_outputs)")
    args = ap.parse_args()
    VETORIZACAO = args.vetorizacao
    SAIDA = Path(args.saida_base or RAIZ / "modelos_outputs") / (NOME if VETORIZACAO == "protocolo" else "svm_rayssa")

    SAIDA.mkdir(parents=True, exist_ok=True)
    X, y = carregar(args.entrada)
    ids_dev, ids_te, folds = dividir(y)
    X_dev, y_dev = X[ids_dev], y[ids_dev]
    print(f"Documentos: {len(y)} | Desenvolvimento: {len(ids_dev)} | Teste (reservado): {len(ids_te)}\n")

    configs = [(m, C) for m in METODOS for C in GRADE_C]
    oof = {c: np.zeros(len(y_dev)) for c in configs}
    por_fold = {c: [] for c in configs}
    t0 = time.time()
    for k in range(5):
        tr, va = folds != k, folds == k
        tfidf = vetorizador()
        Xtr = tfidf.fit_transform(X_dev[tr])      # ajustado só no treino do fold
        Xva = tfidf.transform(X_dev[va])
        for (metodo, C) in configs:
            cal = svm_calibrado(C, metodo).fit(Xtr, y_dev[tr])
            oof[(metodo, C)][va] = cal.predict_proba(Xva)[:, 1]
            por_fold[(metodo, C)].append(metricas_fold(y_dev[va], oof[(metodo, C)][va]))
        print(f"fold {k + 1}/5 concluído ({time.time() - t0:.0f}s)")

    # --------------------------------------------- comparação de todas as configs
    linhas = []
    for (metodo, C) in configs:
        tab = pd.DataFrame(por_fold[(metodo, C)])
        linhas.append({"metodo": metodo, "C": C, **{k: v for k, v in tab.mean().items()},
                       "roc_auc_dp": tab["roc_auc"].std(ddof=1), "brier_dp": tab["brier"].std(ddof=1)})
    comp = pd.DataFrame(linhas).sort_values("brier").reset_index(drop=True)
    comp.to_csv(SAIDA / "comparacao_oof_svm.csv", index=False)
    melhor = comp.groupby("metodo").first()
    print("\nMelhor Brier OOF por método de calibração:")
    print(melhor[["C", "brier", "roc_auc", "pr_auc", "ece"]].round(4).to_string())

    b_sig, b_iso = melhor.loc["sigmoid", "brier"], melhor.loc["isotonic", "brier"]
    if b_iso < b_sig - TOLERANCIA_BRIER:
        escolhido = "isotonic"
        motivo = f"isotonic tem Brier {b_sig - b_iso:.4f} menor (> tolerância {TOLERANCIA_BRIER})"
    else:
        escolhido = "sigmoid"
        motivo = ("sigmoid tem o menor Brier" if b_sig <= b_iso else
                  f"diferença de Brier ({b_sig - b_iso:+.4f}) dentro da tolerância {TOLERANCIA_BRIER}: fica o sigmoid, mais estável")
    C_final = float(melhor.loc[escolhido, "C"])
    print(f"\n>> Configuração escolhida: {escolhido}, C={C_final} — {motivo}")

    oof_melhores = [(m, float(melhor.loc[m, "C"]), oof[(m, float(melhor.loc[m, "C"]))]) for m in METODOS]

    resultado = {
        "modelo": "LinearSVC + CalibratedClassifierCV (TF-IDF)",
        "classe_positiva": "Fake",
        "entrada": args.entrada,
        "vetorizacao": args.vetorizacao,
        "escolhido": escolhido,
        "configuracao_escolhida": {"metodo": escolhido, "C": C_final, "motivo": motivo},
        "n_amostras_validacao_cruzada": int(len(y_dev)),
        "validacao": "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42, grupos = URL) sobre os 80% de desenvolvimento",
        "observacao": "C e método escolhidos pelo menor Brier OOF no próprio desenvolvimento "
                      "(seleção leve; o teste final não participa da escolha).",
    }
    for metodo, C, p in oof_melhores:
        tab = pd.DataFrame(por_fold[(metodo, C)])
        resultado[metodo] = {
            "C": C,
            "media": tab.mean().to_dict(),
            "desvio_padrao": tab.std(ddof=1).to_dict(),
            "roc_auc_ic95": ic95_auc(y_dev, p),
            "pct_previsoes_extremas": float(np.mean((p < 0.01) | (p > 0.99))),
        }
    (SAIDA / "metricas.json").write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")

    with open(SAIDA / "classification_report.txt", "w", encoding="utf-8") as f:
        for metodo, C, p in oof_melhores:
            f.write(f"===== SVM {metodo} C={C} (previsões fora do fold, limiar 0,5) =====\n")
            f.write(classification_report(y_dev, (p >= 0.5).astype(int), target_names=CLASSES, digits=4) + "\n")
    salvar_reliability(y_dev, oof_melhores, escolhido)
    salvar_matriz(y_dev, oof_melhores)

    # --------------------------------------------- saídas para comparação / stacking
    col = f"p_fake_{NOME}"
    pd.DataFrame({"id": ids_dev, "fold": folds, "y_fake": y_dev, col: oof[(escolhido, C_final)]}).to_csv(
        SAIDA / f"oof_{NOME}.csv", index=False)
    if not args.sem_teste:
        tfidf = vetorizador()
        Xd = tfidf.fit_transform(X_dev)
        cal = svm_calibrado(C_final, escolhido).fit(Xd, y_dev)
        pd.DataFrame({"id": ids_te, "y_fake": y[ids_te],
                      col: cal.predict_proba(tfidf.transform(X[ids_te]))[:, 1]}).to_csv(
            SAIDA / f"teste_{NOME}.csv", index=False)

    print("\n=== SVM calibrado (média ± dp entre folds) ===")
    for metodo in METODOS:
        m, s = resultado[metodo]["media"], resultado[metodo]["desvio_padrao"]
        lo, hi = resultado[metodo]["roc_auc_ic95"]
        print(f"  {metodo:8s} C={resultado[metodo]['C']:<5} AUC={m['roc_auc']:.4f}±{s['roc_auc']:.4f} "
              f"IC95=[{lo:.4f},{hi:.4f}]  Brier={m['brier']:.4f}±{s['brier']:.4f}  ECE={m['ece']:.4f}")
    print(f"\nSaídas em {SAIDA}")


if __name__ == "__main__":
    main()