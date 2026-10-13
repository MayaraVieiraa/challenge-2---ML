# -*- coding: utf-8 -*-
"""
treinar_nb.py — Multinomial Naive Bayes calibrado (Rafaella)
============================================================
Script da Rafaella integrado ao projeto principal. A lógica de modelagem dela foi mantida
(TF-IDF + MultinomialNB; CalibratedClassifierCV com sigmoid e isotonic, cv=5, reportando o
Brier ANTES e DEPOIS da calibração). Ajustes feitos na integração:

    * caminhos relativos à raiz do projeto (roda de qualquer pasta);
    * uso do CSV OFICIAL (dados_tratados/NoRemovalWords_tratada_dedup.csv): o CSV local
      que ela usou não era o oficial, então os números dela devem ser refeitos aqui;
    * split explícito e idêntico ao dos outros modelos (ids = posição da linha no CSV),
      para gerar oof_nb.csv / teste_nb.csv comparáveis (stacking);
    * escolha da versão calibrada (sigmoid x isotonic) pelo menor Brier OOF, com o mesmo
      desempate pró-sigmoid usado no SVM;
    * métricas extras: log_loss e ECE.

Protocolo: base NoRemovalWords_tratada_dedup | Titulo + Noticia | TF-IDF padronizado
(1,2)-gramas, min_df=5, max_df=0.9, sublinear_tf, strip_accents="unicode" |
StratifiedKFold(5, shuffle, 42) sobre os 80% de desenvolvimento | teste 80/20 reservado.
Classe positiva = Fake.

Saídas em modelos_outputs/nb/:
    metricas.json, reliability.png, matriz_confusao.png, classification_report.txt,
    oof_nb.csv, teste_nb.csv
Uso:
    python modelos/treinar_nb.py
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
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _divisao import dividir_grupos, grupos_de  # noqa: E402
from _entrada import ENTRADAS, montar_texto  # noqa: E402

SEED = 42
RAIZ = Path(__file__).resolve().parents[1]
DADOS = RAIZ / "dados_tratados" / os.environ.get("BASE_DADOS", "FakeRecogna2_extrativa_casada.csv")
NOME = "nb"
SAIDA = RAIZ / "modelos_outputs" / NOME
CLASSES = ["Real", "Fake"]  # 0 = Real, 1 = Fake (classe positiva)
N_BOOT = 1000
TOLERANCIA_BRIER = 0.0005


# ------------------------------------------------------------------ dados / split
def carregar(entrada="noticia"):
    df = pd.read_csv(DADOS)
    texto = pd.Series(montar_texto(df, entrada))
    y = (df["Rotulo"].astype(str).str.strip().str.lower() == "fake").astype(int).to_numpy()
    return texto.to_numpy(), y


def dividir(y):
    return dividir_grupos(y, grupos_de(DADOS, len(y)))


# ------------------------------------------------------------------ modelos
def vetorizador():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.9,
                           sublinear_tf=True, strip_accents="unicode")


def nb_cru():
    return Pipeline([("tfidf", vetorizador()), ("nb", MultinomialNB())])


MODELOS = {
    "antes_calibracao": nb_cru,
    "depois_sigmoid": lambda: CalibratedClassifierCV(nb_cru(), method="sigmoid", cv=5),
    "depois_isotonic": lambda: CalibratedClassifierCV(nb_cru(), method="isotonic", cv=5),
}


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
def salvar_reliability(y, oof):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], "--", c="gray", label="Calibração perfeita")
    for nome, p in oof.items():
        real, prevista = calibration_curve(y, p, n_bins=10)
        ax.plot(prevista, real, "o-", label=f"{nome} (Brier {brier_score_loss(y, p):.4f})")
    ax.set_xlabel("Probabilidade prevista de Fake")
    ax.set_ylabel("Fração observada de Fake")
    ax.set_title("Multinomial NB: reliability diagram (OOF, 5 folds)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(SAIDA / "reliability.png", dpi=150)
    plt.close(fig)


def salvar_matrizes(y, oof):
    fig, axs = plt.subplots(1, len(oof), figsize=(5 * len(oof), 4.5))
    for ax, (nome, p) in zip(axs, oof.items()):
        cm = confusion_matrix(y, (p >= 0.5).astype(int), labels=[0, 1])
        ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, colorbar=False)
        ax.set_title(nome)
    fig.tight_layout()
    fig.savefig(SAIDA / "matriz_confusao.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    global SAIDA
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sem-teste", action="store_true")
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
    print(f"Documentos: {len(y)} | Desenvolvimento: {len(ids_dev)} | Teste (reservado): {len(ids_te)}")

    por_fold = {nome: [] for nome in MODELOS}
    oof = {nome: np.zeros(len(y_dev)) for nome in MODELOS}
    t0 = time.time()
    for k in range(5):
        tr, va = folds != k, folds == k
        for nome, fabrica in MODELOS.items():
            modelo = fabrica().fit(X_dev[tr], y_dev[tr])
            oof[nome][va] = modelo.predict_proba(X_dev[va])[:, 1]
            por_fold[nome].append(metricas_fold(y_dev[va], oof[nome][va]))
        print(f"fold {k + 1}/5 concluído ({time.time() - t0:.0f}s)")

    resultado = {
        "modelo": "MultinomialNB + TF-IDF",
        "classe_positiva": "Fake",
        "entrada": args.entrada,
        "n_amostras_validacao_cruzada": int(len(y_dev)),
        "validacao": "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42, grupos = URL) sobre os 80% de desenvolvimento",
    }
    for nome in MODELOS:
        tabela = pd.DataFrame(por_fold[nome])
        p = oof[nome]
        resultado[nome] = {
            "media": tabela.mean().to_dict(),
            "desvio_padrao": tabela.std(ddof=1).to_dict(),
            "roc_auc_ic95": ic95_auc(y_dev, p),
            "pct_previsoes_extremas": float(np.mean((p < 0.01) | (p > 0.99))),
        }
    resultado["ganho_brier_vs_antes"] = {
        n: resultado["antes_calibracao"]["media"]["brier"] - resultado[n]["media"]["brier"]
        for n in ("depois_sigmoid", "depois_isotonic")}

    b_sig = resultado["depois_sigmoid"]["media"]["brier"]
    b_iso = resultado["depois_isotonic"]["media"]["brier"]
    escolhido = "depois_isotonic" if b_iso < b_sig - TOLERANCIA_BRIER else "depois_sigmoid"
    resultado["escolhido"] = escolhido
    resultado["criterio_escolha"] = (f"menor Brier OOF; empate (< {TOLERANCIA_BRIER}) fica o sigmoid. "
                                     f"Brier sigmoid={b_sig:.4f}, isotonic={b_iso:.4f}")
    (SAIDA / "metricas.json").write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")

    with open(SAIDA / "classification_report.txt", "w", encoding="utf-8") as f:
        for nome, p in oof.items():
            f.write(f"===== {nome} (previsões fora do fold, limiar 0,5) =====\n")
            f.write(classification_report(y_dev, (p >= 0.5).astype(int), target_names=CLASSES, digits=4) + "\n")
    salvar_reliability(y_dev, oof)
    salvar_matrizes(y_dev, oof)

    col = f"p_fake_{NOME}"
    pd.DataFrame({"id": ids_dev, "fold": folds, "y_fake": y_dev, col: oof[escolhido]}).to_csv(
        SAIDA / f"oof_{NOME}.csv", index=False)
    if not args.sem_teste:
        final = MODELOS[escolhido]().fit(X_dev, y_dev)
        pd.DataFrame({"id": ids_te, "y_fake": y[ids_te], col: final.predict_proba(X[ids_te])[:, 1]}).to_csv(
            SAIDA / f"teste_{NOME}.csv", index=False)

    print(f"\n{'':18}{'brier':>16}{'roc_auc':>16}{'pr_auc':>16}")
    for nome in MODELOS:
        m, s = resultado[nome]["media"], resultado[nome]["desvio_padrao"]
        print(f"{nome:18}" + "".join(f"{m[c]:>9.4f} ±{s[c]:.4f}" for c in ("brier", "roc_auc", "pr_auc")))
    print(f"\nEscolhido: {escolhido} ({resultado['criterio_escolha']})")
    print(f"Saídas em {SAIDA}")


if __name__ == "__main__":
    main()