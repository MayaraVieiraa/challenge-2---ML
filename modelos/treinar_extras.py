# -*- coding: utf-8 -*-
"""
treinar_extras.py — dois modelos EXTRAS e diversos para o stacking
==================================================================
O stacking só ajuda se os modelos base errarem de formas diferentes. Baseline, SVM e NB usam
o mesmo tipo de feature (TF-IDF de palavras); estes dois mudam a representação/o algoritmo:

  char : TF-IDF de CARACTERES (n-gramas 3-5, char_wb) + LinearSVC + CalibratedClassifierCV
         (sigmoid, cv=5). Pega morfologia, grafia, pontuação, URLs/emojis — estilo, não vocabulário.
  gbm  : TF-IDF de palavras -> TruncatedSVD(200) -> HistGradientBoosting. Interações não lineares
         em um espaço semântico latente (calibração nativa).

Mesmo protocolo dos demais: split idêntico (teste 80/20 + 5 folds, seed 42), mesma --entrada,
classe positiva = Fake, metricas.json, oof_<modelo>.csv, teste_<modelo>.csv (sem métrica de teste).
Saídas: modelos_outputs/char/ e modelos_outputs/gbm/.
Uso:  python modelos/treinar_extras.py [--so char|gbm] [--entrada ...]
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
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, classification_report,
                             log_loss, precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _divisao import dividir_grupos, grupos_de  # noqa: E402
from _entrada import ENTRADAS, montar_texto  # noqa: E402

SEED = 42
RAIZ = Path(__file__).resolve().parents[1]
DADOS = RAIZ / "dados_tratados" / os.environ.get("BASE_DADOS", "FakeRecogna2_extrativa_casada.csv")
CLASSES = ["Real", "Fake"]


def modelo_char():
    return Pipeline([("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_df=0.9,
                                               sublinear_tf=True, max_features=300_000, dtype=np.float32)),
                     ("clf", CalibratedClassifierCV(LinearSVC(C=1.0, dual="auto", max_iter=20_000, random_state=SEED),
                                                    method="sigmoid", cv=StratifiedKFold(5, shuffle=True, random_state=SEED)))])


def modelo_gbm():
    return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 1), min_df=5, max_df=0.9, sublinear_tf=True,
                                               strip_accents="unicode", max_features=30_000)),
                     ("svd", TruncatedSVD(200, random_state=SEED)),
                     ("gbm", HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=250,
                                                            l2_regularization=1.0, random_state=SEED))])


MODELOS = {"char": modelo_char, "gbm": modelo_gbm}


def ece(y, p, bins=10):
    e = np.linspace(0, 1, bins + 1)
    i = np.clip(np.digitize(p, e[1:-1]), 0, bins - 1)
    return float(sum((i == b).mean() * abs(y[i == b].mean() - p[i == b].mean()) for b in range(bins) if (i == b).any()))


def metricas(y, p):
    pred = (p >= 0.5).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    m = {"accuracy": accuracy_score(y, pred), "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
         "brier": brier_score_loss(y, p), "log_loss": log_loss(y, np.clip(p, 1e-15, 1 - 1e-15)), "ece": ece(y, p)}
    for i, c in enumerate(CLASSES):
        m[f"precision_{c}"], m[f"recall_{c}"], m[f"f1_{c}"] = pr[i], rc[i], f1[i]
    return m


def ic95(y, p, n=1000):
    rng = np.random.default_rng(SEED)
    a = []
    while len(a) < n:
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            a.append(roc_auc_score(y[i], p[i]))
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--so", choices=list(MODELOS), default=None)
    ap.add_argument("--entrada", choices=ENTRADAS, default="noticia")
    ap.add_argument("--saida-base", default=None)
    ap.add_argument("--sem-teste", action="store_true")
    a = ap.parse_args()
    base = Path(a.saida_base or RAIZ / "modelos_outputs")

    df = pd.read_csv(DADOS)
    X = montar_texto(df, a.entrada)
    y = (df["Rotulo"].astype(str).str.strip().str.lower() == "fake").astype(int).to_numpy()
    ids_dev, ids_te, folds = dividir_grupos(y, grupos_de(DADOS, len(y)))
    Xd, yd = X[ids_dev], y[ids_dev]

    for nome, fab in MODELOS.items():
        if a.so and nome != a.so:
            continue
        saida = base / nome
        saida.mkdir(parents=True, exist_ok=True)
        oof, pf, t0 = np.zeros(len(yd)), [], time.time()
        for k in range(5):
            tr, va = folds != k, folds == k
            oof[va] = fab().fit(Xd[tr], yd[tr]).predict_proba(Xd[va])[:, 1]
            pf.append(metricas(yd[va], oof[va]))
            print(f"[{nome}] fold {k + 1}/5 AUC={pf[-1]['roc_auc']:.4f} Brier={pf[-1]['brier']:.4f} ({time.time() - t0:.0f}s)", flush=True)
        t = pd.DataFrame(pf)
        res = {"modelo": nome, "classe_positiva": "Fake", "entrada": a.entrada, "escolhido": nome,
               "n_amostras_validacao_cruzada": int(len(yd)),
               "validacao": "StratifiedGroupKFold(5, shuffle=True, random_state=42, grupos = URL) sobre os 80% de desenvolvimento",
               nome: {"media": t.mean().to_dict(), "desvio_padrao": t.std(ddof=1).to_dict(), "roc_auc_ic95": ic95(yd, oof)}}
        (saida / "metricas.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        (saida / "classification_report.txt").write_text(
            classification_report(yd, (oof >= 0.5).astype(int), target_names=CLASSES, digits=4), encoding="utf-8")
        fig, ax = plt.subplots(figsize=(5.5, 5.2))
        ax.plot([0, 1], [0, 1], "--", c="gray")
        r, p_ = calibration_curve(yd, oof, n_bins=10)
        ax.plot(p_, r, "o-", label=f"{nome} (Brier {brier_score_loss(yd, oof):.4f})")
        ax.set(xlabel="P(Fake) prevista", ylabel="Fração observada", title=f"{nome}: reliability (OOF)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(saida / "reliability.png", dpi=140)
        plt.close(fig)
        col = f"p_fake_{nome}"
        pd.DataFrame({"id": ids_dev, "fold": folds, "y_fake": yd, col: oof}).to_csv(saida / f"oof_{nome}.csv", index=False)
        if not a.sem_teste:
            m = fab().fit(Xd, yd)
            pd.DataFrame({"id": ids_te, "y_fake": y[ids_te], col: m.predict_proba(X[ids_te])[:, 1]}).to_csv(
                saida / f"teste_{nome}.csv", index=False)
        mm = res[nome]["media"]
        print(f"== {nome}: AUC={mm['roc_auc']:.4f} IC95={res[nome]['roc_auc_ic95']} Brier={mm['brier']:.4f} ECE={mm['ece']:.4f}", flush=True)


if __name__ == "__main__":
    main()