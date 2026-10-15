# -*- coding: utf-8 -*-
"""
stacking.py — comparação final dos modelos + soft voting + STACKING (etapa 7 do plano)
======================================================================================
Lê as probabilidades fora-do-fold (OOF) gravadas por cada script de modelo em
modelos_outputs/<modelo>/oof_<modelo>.csv, confere que TODOS usaram exatamente o mesmo split
e então:

  1. consolida a tabela comparativa (ROC-AUC com IC95% por bootstrap, PR-AUC, Brier, ECE,
     acurácia só como referência, média ± dp entre folds);
  2. mede a CORRELAÇÃO entre os modelos (stacking só ajuda se os erros forem diferentes);
  3. monta dois ensembles:
       - soft voting: média simples das probabilidades (método padrão do protocolo);
       - stacking: meta-modelo (Regressão Logística sobre o logit das probabilidades dos modelos
         base). O meta-modelo é avaliado em CV com os MESMOS 5 folds: para o fold k ele aprende
         só com as linhas OOF dos outros folds;
  4. aplica o critério de decisão do protocolo (seção 9): (1) maior ROC-AUC com IC95% não
     sobreposto ao baseline; (2) menor Brier; (3) menor complexidade em empate — e o teste de
     "ganho consistente" para ensembles (IC95% do ganho pareado exclui 0 e vitória em >= 4 de 5
     folds, tanto em AUC quanto em Brier);
  5. (opcional, --avaliar-teste) avaliação ÚNICA e FINAL no teste reservado: meta-modelo
     treinado em todo o OOF do desenvolvimento, aplicado às probabilidades de teste dos modelos
     base (que foram ajustados em todo o desenvolvimento). Só rode isto depois de fixar o
     modelo pelo critério acima — o teste não pode orientar escolhas.

Cuidado conhecido do stacking: as probabilidades OOF de treino do meta-modelo vêm de modelos que
viram os dados do fold avaliado; é o procedimento padrão e o viés é pequeno, mas por isso o
protocolo (seção 9) pede ganho CONSISTENTE antes de adotar um ensemble.

Saídas em modelos_outputs/stacking/.
Uso:
    python modelos/stacking.py                  # só validação cruzada
    python modelos/stacking.py --avaliar-teste  # + avaliação final no teste
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score,
                             brier_score_loss, classification_report, confusion_matrix, roc_auc_score)

SEED = 42
RAIZ = Path(__file__).resolve().parents[1]
MODELOS_BASE = ["baseline", "svm", "nb", "char", "gbm", "bertopic"]
ROTULOS = {"baseline": "LR (baseline)", "svm": "SVM calibrado", "nb": "NB calibrado", "bertopic": "BERTopic+LR", "char": "Char n-gramas+SVM", "gbm": "SVD+GBM",
           "soft_voting": "Soft voting", "stacking": "Stacking (LR meta)"}
COMPLEXIDADE = {"baseline": 1, "nb": 1, "svm": 2, "char": 3, "gbm": 4, "bertopic": 4, "soft_voting": 5, "stacking": 6}
TOL_BRIER = 0.0005
N_BOOT = 1000


# ------------------------------------------------------------------ utilidades
def md_tabela(df, indice=True):
    d = df.reset_index() if indice else df
    cols = list(d.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join(out)


def logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def metricas(y, p):
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "ece": ece(y, p), "accuracy": accuracy_score(y, p >= 0.5)}


def ic95(y, p, n=N_BOOT):
    rng = np.random.default_rng(SEED)
    a = []
    while len(a) < n:
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            a.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def por_fold(y, p, folds):
    r = [{"roc_auc": roc_auc_score(y[folds == k], p[folds == k]), "brier": brier_score_loss(y[folds == k], p[folds == k])}
         for k in range(5)]
    return pd.DataFrame(r)


def delta_pareado(y, pa, pb, n=N_BOOT):
    """Bootstrap pareado de (A - B): ganho de AUC e ganho de Brier (positivo = A melhor)."""
    rng = np.random.default_rng(SEED)
    da, db = [], []
    while len(da) < n:
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        da.append(roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i]))
        db.append(brier_score_loss(y[i], pb[i]) - brier_score_loss(y[i], pa[i]))
    f = lambda v: (float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return f(da), f(db)


# ------------------------------------------------------------------ carga e conferência
def carregar_oof(base, modelos):
    tabelas, usados = [], []
    for m in modelos:
        arq = base / m / f"oof_{m}.csv"
        if not arq.exists():
            print(f"  [aviso] {arq} não encontrado — modelo '{m}' fora do stacking")
            continue
        t = pd.read_csv(arq).rename(columns={f"p_fake_{m}": m})
        tabelas.append(t[["id", "fold", "y_fake", m]])
        usados.append(m)
    if len(usados) < 2:
        raise SystemExit("Preciso de pelo menos 2 modelos com oof_*.csv. Rode os scripts de treino antes.")
    ref = tabelas[0]
    for m, t in zip(usados[1:], tabelas[1:]):
        if len(t) != len(ref) or not (t["id"].to_numpy() == ref["id"].to_numpy()).all():
            raise SystemExit(f"'{m}' usa OUTRO conjunto/ordem de documentos — todos os modelos precisam do mesmo split.")
        if not (t["fold"].to_numpy() == ref["fold"].to_numpy()).all():
            raise SystemExit(f"'{m}' usa OUTROS folds — todos os modelos precisam de StratifiedKFold(5, shuffle, 42).")
        if not (t["y_fake"].to_numpy() == ref["y_fake"].to_numpy()).all():
            raise SystemExit(f"'{m}' tem rótulos diferentes — base de dados inconsistente.")
    df = ref[["id", "fold", "y_fake"]].copy()
    for m, t in zip(usados, tabelas):
        df[m] = t[m].to_numpy()
    return df, usados


def carregar_teste(base, usados):
    ts = []
    for m in usados:
        arq = base / m / f"teste_{m}.csv"
        if not arq.exists():
            raise SystemExit(f"{arq} não existe (rode o modelo '{m}' sem --sem-teste).")
        ts.append(pd.read_csv(arq).rename(columns={f"p_fake_{m}": m}))
    ref = ts[0]
    for m, t in zip(usados[1:], ts[1:]):
        if not (t["id"].to_numpy() == ref["id"].to_numpy()).all():
            raise SystemExit(f"'{m}': documentos de teste diferentes dos outros modelos.")
    out = ref[["id", "y_fake"]].copy()
    for m, t in zip(usados, ts):
        out[m] = t[m].to_numpy()
    return out


# ------------------------------------------------------------------ stacking
def novo_meta(C=1.0):
    return LogisticRegression(C=C, max_iter=2000, random_state=SEED)


def stacking_cv(df, usados, C=1.0):
    """Meta-modelo avaliado nos mesmos folds: no fold k aprende só com os outros folds."""
    Z = np.column_stack([logit(df[m].to_numpy()) for m in usados])
    y, folds = df["y_fake"].to_numpy(), df["fold"].to_numpy()
    oof = np.zeros(len(y))
    coefs = []
    for k in range(5):
        tr, va = folds != k, folds == k
        meta = novo_meta(C).fit(Z[tr], y[tr])
        oof[va] = meta.predict_proba(Z[va])[:, 1]
        coefs.append(np.r_[meta.coef_.ravel(), meta.intercept_])
    return oof, np.array(coefs)


# ------------------------------------------------------------------ gráficos
def grafico_correlacao(df, usados, saida):
    Z = pd.DataFrame({ROTULOS[m]: logit(df[m].to_numpy()) for m in usados})
    corr = Z.corr()
    erro = pd.DataFrame({ROTULOS[m]: ((df[m] >= 0.5).astype(int) != df["y_fake"]).astype(int) for m in usados})
    corr_erro = erro.corr()
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, c, tit in ((axs[0], corr, "Correlação dos logits (P(Fake))"), (axs[1], corr_erro, "Correlação dos ERROS (0,5)")):
        im = ax.imshow(c.values, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(c)), c.columns, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(c)), c.columns, fontsize=8)
        for i in range(len(c)):
            for j in range(len(c)):
                ax.text(j, i, f"{c.values[i, j]:.2f}", ha="center", va="center", color="w" if c.values[i, j] < .6 else "k", fontsize=8)
        ax.set_title(tit)
    fig.tight_layout()
    fig.savefig(saida / "correlacao_modelos.png", dpi=140)
    plt.close(fig)
    corr.to_csv(saida / "correlacao_logits.csv", encoding="utf-8")
    corr_erro.to_csv(saida / "correlacao_erros.csv", encoding="utf-8")
    return corr, corr_erro


def grafico_reliability(y, probs, saida):
    fig, ax = plt.subplots(figsize=(6.6, 6))
    ax.plot([0, 1], [0, 1], "--", c="gray", label="Calibração perfeita")
    for nome, p in probs.items():
        real, prev = calibration_curve(y, p, n_bins=10)
        ax.plot(prev, real, "o-", ms=4, label=f"{ROTULOS.get(nome, nome)} (Brier {brier_score_loss(y, p):.4f})")
    ax.set(xlabel="Probabilidade prevista de Fake", ylabel="Fração observada de Fake",
           title="Reliability diagram — modelos base e ensembles (OOF)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(saida / "reliability.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ decisão
def escolher(tab, baseline="baseline"):
    """Critério do protocolo, seção 9."""
    hi_base = tab.loc[baseline, "auc_ic95_sup"] if baseline in tab.index else np.inf
    cand = tab[(tab["auc_ic95_inf"] > hi_base) & (tab.index != baseline)]
    passo1 = len(cand) > 0
    pool = cand if passo1 else tab
    b = pool["brier"].min()
    empate = pool[pool["brier"] <= b + TOL_BRIER]
    venc = min(empate.index, key=lambda m: (COMPLEXIDADE.get(m, 9), pool.loc[m, "brier"]))
    return venc, passo1, list(empate.index)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--saida-base", default=None, help="pasta com os modelos (padrão: <raiz>/modelos_outputs)")
    ap.add_argument("--modelos", nargs="+", default=MODELOS_BASE)
    ap.add_argument("--C-meta", type=float, default=1.0)
    ap.add_argument("--avaliar-teste", action="store_true", help="avaliação final única no teste reservado")
    args = ap.parse_args()
    base = Path(args.saida_base or RAIZ / "modelos_outputs")
    saida = base / "stacking"
    saida.mkdir(parents=True, exist_ok=True)

    df, usados = carregar_oof(base, args.modelos)
    y, folds = df["y_fake"].to_numpy(), df["fold"].to_numpy()
    print(f"Modelos base: {', '.join(usados)} | linhas OOF: {len(df)}")

    probs = {m: df[m].to_numpy() for m in usados}
    probs["soft_voting"] = np.mean([probs[m] for m in usados], axis=0)
    probs["stacking"], coefs_cv = stacking_cv(df, usados, args.C_meta)

    linhas, pf = [], {}
    for nome, p in probs.items():
        lo, hi = ic95(y, p)
        f = por_fold(y, p, folds)
        pf[nome] = f
        linhas.append({"modelo": nome, **metricas(y, p), "auc_ic95_inf": lo, "auc_ic95_sup": hi,
                       "auc_media_folds": f.roc_auc.mean(), "auc_dp_folds": f.roc_auc.std(ddof=1),
                       "brier_media_folds": f.brier.mean(), "brier_dp_folds": f.brier.std(ddof=1)})
    tab = pd.DataFrame(linhas).set_index("modelo")
    tab.to_csv(saida / "comparacao_modelos_cv.csv", encoding="utf-8")
    print("\n" + tab[["roc_auc", "auc_ic95_inf", "auc_ic95_sup", "pr_auc", "brier", "ece", "accuracy"]].round(4).to_string())

    corr, corr_erro = grafico_correlacao(df, usados, saida)
    grafico_reliability(y, probs, saida)

    # meta-modelo final (todo o OOF) e pesos
    Z = np.column_stack([logit(df[m].to_numpy()) for m in usados])
    meta = novo_meta(args.C_meta).fit(Z, y)
    pesos = pd.DataFrame({"modelo": usados + ["intercepto"],
                          "peso_meta_final": np.r_[meta.coef_.ravel(), meta.intercept_],
                          "peso_medio_folds": coefs_cv.mean(axis=0), "peso_dp_folds": coefs_cv.std(axis=0, ddof=1)})
    pesos.to_csv(saida / "pesos_meta.csv", index=False, encoding="utf-8")

    # ---------------- ganho consistente dos ensembles
    individuais = [m for m in usados]
    melhor_ind = min(individuais, key=lambda m: (tab.loc[m, "brier"]))
    melhor_auc = max(individuais, key=lambda m: tab.loc[m, "roc_auc"])
    ganhos = []
    for ens in ("soft_voting", "stacking"):
        for ref in sorted({melhor_ind, melhor_auc, "baseline"} & set(usados)):
            (da, da_lo, da_hi), (db, db_lo, db_hi) = delta_pareado(y, probs[ens], probs[ref])
            v_auc = int((pf[ens].roc_auc.to_numpy() > pf[ref].roc_auc.to_numpy()).sum())
            v_brier = int((pf[ens].brier.to_numpy() < pf[ref].brier.to_numpy()).sum())
            consistente = (da_lo > 0) and (db_lo > 0) and v_auc >= 4 and v_brier >= 4
            ganhos.append({"ensemble": ens, "contra": ref, "ganho_auc": da, "auc_ic95_inf": da_lo, "auc_ic95_sup": da_hi,
                           "ganho_brier": db, "brier_ic95_inf": db_lo, "brier_ic95_sup": db_hi,
                           "folds_melhor_auc": f"{v_auc}/5", "folds_melhor_brier": f"{v_brier}/5",
                           "ganho_consistente": "SIM" if consistente else "não"})
    ganhos = pd.DataFrame(ganhos)
    ganhos.to_csv(saida / "ganho_ensemble_pareado.csv", index=False, encoding="utf-8")
    print("\nGanho pareado dos ensembles (positivo = ensemble melhor):")
    print(ganhos.round(4).to_string(index=False))

    venc_modelos, p1, empatados = escolher(tab.loc[usados], "baseline")
    venc_todos, p1_t, empatados_t = escolher(tab, "baseline")
    cons = ganhos[ganhos.ganho_consistente == "SIM"]
    decisao_final = venc_todos
    if venc_todos in ("soft_voting", "stacking") and not len(cons[cons.ensemble == venc_todos]):
        decisao_final = venc_modelos        # ensemble sem ganho consistente -> volta ao melhor individual
    # ---------------- relatório da decisão
    md = ["# Decisão de modelo — critério do protocolo (seção 9)", "",
          "Critérios em ordem: (1) maior ROC-AUC com IC95% **não sobreposto** ao baseline; (2) menor Brier; "
          f"(3) menor complexidade em empate (|ΔBrier| ≤ {TOL_BRIER}). Acurácia não entra. Ensemble só se houver ganho **consistente**.", "",
          "## Tabela (OOF, 5 folds do protocolo)", "",
          tab[["roc_auc", "auc_ic95_inf", "auc_ic95_sup", "pr_auc", "brier", "ece", "accuracy"]].round(4).pipe(md_tabela, True), "",
          "## Ganho pareado dos ensembles", "", ganhos.round(4).pipe(md_tabela, False), "",
          "## Pesos do meta-modelo (logit das probabilidades)", "", pesos.round(3).pipe(md_tabela, False), "",
          "## Resultado", "",
          f"- Entre os modelos individuais: **{ROTULOS[venc_modelos]}** "
          + ("(IC95% de AUC não sobreposto ao do baseline)." if p1 else
             "(nenhum IC95% de AUC fica fora do baseline: empate no critério 1, decide o Brier e depois a simplicidade)."),
          f"- Incluindo ensembles: **{ROTULOS[venc_todos]}**; decisão final após o teste de ganho consistente: **{ROTULOS[decisao_final]}**.", ""]
    (saida / "decisao_modelo.md").write_text("\n".join(md), encoding="utf-8")

    # ---------------- saídas do stacking (formato do protocolo)
    f = pf["stacking"]
    res = {"modelo": "Stacking: LogisticRegression sobre logit(P(Fake)) dos modelos base",
           "classe_positiva": "Fake", "escolhido": "stacking", "modelos_base": usados,
           "n_amostras_validacao_cruzada": int(len(y)),
           "validacao": "meta-modelo avaliado nos mesmos 5 folds (aprende no OOF dos outros folds)",
           "stacking": {"media": {**f.mean().to_dict(), **{k: float(tab.loc['stacking', k]) for k in ('pr_auc', 'ece', 'accuracy')}},
                        "desvio_padrao": f.std(ddof=1).to_dict(),
                        "roc_auc_ic95": [tab.loc["stacking", "auc_ic95_inf"], tab.loc["stacking", "auc_ic95_sup"]]},
           "decisao": {"melhor_individual": venc_modelos, "melhor_com_ensembles": venc_todos, "final": decisao_final}}
    (saida / "metricas.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    (saida / "classification_report.txt").write_text(
        "===== Stacking (previsões fora do fold, limiar 0,5) =====\n" +
        classification_report(y, (probs["stacking"] >= 0.5).astype(int), target_names=["Real", "Fake"], digits=4), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    ConfusionMatrixDisplay(confusion_matrix(y, (probs["stacking"] >= 0.5).astype(int), labels=[0, 1]),
                           display_labels=["Real", "Fake"]).plot(ax=ax, colorbar=False)
    ax.set_title("Stacking — matriz de confusão (OOF)")
    fig.tight_layout()
    fig.savefig(saida / "matriz_confusao.png", dpi=150)
    plt.close(fig)
    pd.DataFrame({"id": df["id"], "fold": folds, "y_fake": y, "p_fake_stacking": probs["stacking"],
                  "p_fake_soft_voting": probs["soft_voting"]}).to_csv(saida / "oof_stacking.csv", index=False)

    print(f"\nIndividual: {venc_modelos} | com ensembles: {venc_todos} | decisão final: {decisao_final}")
    print("Pesos do meta-modelo (final):", dict(zip(pesos.modelo, pesos.peso_meta_final.round(3))))

    # ---------------- teste final (opcional)
    if args.avaliar_teste:
        te = carregar_teste(base, usados)
        yt = te["y_fake"].to_numpy()
        pt = {m: te[m].to_numpy() for m in usados}
        pt["soft_voting"] = np.mean([pt[m] for m in usados], axis=0)
        pt["stacking"] = meta.predict_proba(np.column_stack([logit(pt[m]) for m in usados]))[:, 1]
        linhas = []
        for nome, p in pt.items():
            lo, hi = ic95(yt, p)
            linhas.append({"modelo": nome, **metricas(yt, p), "auc_ic95_inf": lo, "auc_ic95_sup": hi})
        tt = pd.DataFrame(linhas).set_index("modelo")
        tt.to_csv(saida / "teste_final_metricas.csv", encoding="utf-8")
        pd.DataFrame({"id": te["id"], "y_fake": yt, **{f"p_fake_{k}": v for k, v in pt.items()}}).to_csv(
            saida / "teste_stacking.csv", index=False)
        (saida / "teste_final_LEIA-ME.txt").write_text(
            f"AVALIAÇÃO ÚNICA NO TESTE RESERVADO ({len(yt)} notícias). Feita DEPOIS de fixar o modelo pela validação cruzada "
            f"(decisão final: {decisao_final}). Não use estes números para escolher outro modelo.\n", encoding="utf-8")
        print("\n=== TESTE FINAL (avaliação única) ===")
        print(tt[["roc_auc", "auc_ic95_inf", "auc_ic95_sup", "pr_auc", "brier", "ece", "accuracy"]].round(4).to_string())
    else:
        print("\n(teste final NÃO avaliado — use --avaliar-teste quando o modelo estiver fixado)")
    print(f"\nSaídas em {saida}")


if __name__ == "__main__":
    main()
