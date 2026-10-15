# -*- coding: utf-8 -*-
"""
modelos/avaliar_externo.py

Teste EXTERNO: treina cada modelo SO nos ~80% de desenvolvimento da base de treino (o teste reservado
continua intocado) e mede o desempenho num CSV de outra origem (ex.: Fake.br, de preparar_fakebr.py).
Se o desempenho cair muito em relacao a validacao cruzada (~0,97 a 0,98 de AUC), os modelos
estavam aprendendo caracteristicas desta colecao, e nao a distinguir noticia falsa de verdadeira.

Uso (raiz do projeto):
    python3 modelos/avaliar_externo.py --externo dados_externos/fakebr_t100.csv
    python3 modelos/avaliar_externo.py --externo dados_externos/fakebr_t100.csv --salvar

O CSV externo precisa das colunas Noticia e Rotulo (Fake/Real); Dominio e opcional.
Saidas em modelos_outputs_<base>/externo_<nome>/ : metricas.csv, predicoes.csv
Com --salvar, grava cada modelo treinado em modelos_salvos/<base>/<modelo>.joblib.

Alem dos modelos individuais, avalia o STACKING (mesmo desenho do stacking.py: meta-modelo
logistico sobre o logit das probabilidades fora-da-dobra do desenvolvimento). Use
--sem-stacking para pular essa parte (ela refaz o treino de cada modelo em 5 dobras).

--treino-externo F (0 a 1): poe a fracao F dos PARES do CSV externo no TREINO (junto com a base) e
avalia so nos pares restantes. Cada par (fake_N + true_N, mesma materia) fica inteiro de um lado, e a
divisao e fixa (seed 42). Motivo: a base de treino tem Real quase so de 2020-2021 (covid) e Fake de
varias epocas; sem ver noticias de outra epoca/fonte, o modelo marca como Fake 26% dos Reais do
Fake.br. Com F=0,5 (stacking, pares de teste) o AUC no Fake.br sobe de 0,74 para 0,96 e os Reais marcados como
Fake (limiar 0,5) caem de 26% para 10%. O teste deixa de ser independente: os pares de teste vem das
mesmas fontes dos de treino. Os modelos
salvos vao para modelos_salvos/<base>_com_fakebr/ (o classificar.py usa essa pasta se ela existir).

Tambem mostra o AUC SEM o site dominante dos Fake (ex.: diariodobrasil.org, 92% dos Fake do
Fake.br), para ver se o resultado depende de um unico site.
"""
import argparse
import importlib
import os
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")


def construtores():
    m = {}
    b = importlib.import_module("treinar_baseline")
    s = importlib.import_module("treinar_svm")
    n = importlib.import_module("treinar_nb")
    e = importlib.import_module("treinar_extras")
    m["baseline"] = b.montar_modelo
    m["svm"] = lambda: Pipeline([("tfidf", s.vetorizador()), ("clf", s.svm_calibrado(1.0, "sigmoid"))])
    m["nb"] = n.MODELOS["depois_sigmoid"]   # o escolhido na validacao cruzada (ver nb/metricas.json)
    m["char"] = e.MODELOS["char"]
    m["gbm"] = e.MODELOS["gbm"]
    return m


def logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def metricas(nome, ye, p):
    pred = (p >= 0.5).astype(int)
    return {"modelo": nome, "roc_auc": roc_auc_score(ye, p), "accuracy": accuracy_score(ye, pred),
            "recall_fake": float(pred[ye == 1].mean()), "recall_real": float(1 - pred[ye == 0].mean()),
            "p_fake_media_fake": float(p[ye == 1].mean()), "p_fake_media_real": float(p[ye == 0].mean()),
            "brier": brier_score_loss(ye, p)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--externo", required=True)
    ap.add_argument("--base", default=os.environ.get("BASE_DADOS", "FakeRecogna2_extrativa_casada.csv"))
    ap.add_argument("--modelos", nargs="+", default=["baseline", "svm", "nb", "char", "gbm"])
    ap.add_argument("--salvar", action="store_true", help="grava os modelos treinados (joblib)")
    ap.add_argument("--treino-externo", type=float, default=0.0, metavar="F",
                    help="fracao dos pares do externo que entra no treino (0 = nenhum; 0,5 recomendado)")
    ap.add_argument("--sem-stacking", action="store_true", help="nao avalia o stacking")
    a = ap.parse_args()
    os.environ["BASE_DADOS"] = a.base

    from _divisao import dividir_grupos, grupos_de
    from _entrada import montar_texto
    arq_base = RAIZ / "dados_tratados" / a.base
    df = pd.read_csv(arq_base)
    X = montar_texto(df, "noticia")
    y = (df["Rotulo"].astype(str).str.lower() == "fake").astype(int).to_numpy()
    ids_dev, _, folds = dividir_grupos(y, grupos_de(arq_base, len(y)))
    Xd, yd = X[ids_dev], y[ids_dev]

    ext = pd.read_csv(RAIZ / a.externo if not Path(a.externo).is_absolute() else a.externo)
    Xe = ext["Noticia"].fillna("").astype(str).to_numpy()
    ye = (ext["Rotulo"].astype(str).str.lower() == "fake").astype(int).to_numpy()
    print(f"Treino: {len(yd)} textos (dev de {a.base}) | Externo: {len(ye)} textos "
          f"({ye.mean():.1%} Fake) de {Path(a.externo).name}", flush=True)

    nome_ext = Path(a.externo).stem
    pasta_salvos = Path(a.base).stem
    if a.treino_externo > 0:
        if not 0 < a.treino_externo < 1 or "id" not in ext:
            sys.exit("--treino-externo exige 0 < F < 1 e uma coluna 'id' (fake_N / true_N) no CSV externo")
        par = ext["id"].astype(str).str.split("_").str[-1].to_numpy()
        pares = np.unique(par)
        rng = np.random.default_rng(42)
        treino_par = set(rng.permutation(pares)[:int(round(a.treino_externo * len(pares)))])
        no_treino = np.array([q in treino_par for q in par])
        # folds do trecho externo: por par, para o stacking nao ver o mesmo par no treino e na validacao
        f_ext_all = np.full(int(no_treino.sum()), -1)
        sg = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        for k, (_, va) in enumerate(sg.split(Xe[no_treino], ye[no_treino], groups=par[no_treino])):
            f_ext_all[va] = k
        Xd = np.concatenate([Xd, Xe[no_treino]])
        yd = np.concatenate([yd, ye[no_treino]])
        folds = np.concatenate([folds, f_ext_all])
        ext, Xe, ye = ext[~no_treino].reset_index(drop=True), Xe[~no_treino], ye[~no_treino]
        nome_ext += f"_treino{int(round(a.treino_externo * 100))}"
        pasta_salvos += "_com_fakebr"
        print(f"{int(no_treino.sum())} textos ({len(treino_par)} pares) do externo foram para o TREINO; "
              f"teste externo: {len(ye)} textos ({ye.mean():.1%} Fake). Treino total: {len(yd)}", flush=True)
    saida = RAIZ / f"modelos_outputs_{Path(a.base).stem.replace('FakeRecogna2_', '')}" / f"externo_{nome_ext}"
    saida.mkdir(parents=True, exist_ok=True)
    cons = construtores()
    modelos_ajustados = {}
    linhas, preds = [], pd.DataFrame({"id": ext["id"] if "id" in ext else np.arange(len(ye)), "y_fake": ye})
    for nome in a.modelos:
        t0 = time.time()
        mod = cons[nome]().fit(Xd, yd)
        p = mod.predict_proba(Xe)[:, 1]
        preds[f"p_fake_{nome}"] = p
        linhas.append(metricas(nome, ye, p))
        print(f"[{nome}] AUC={linhas[-1]['roc_auc']:.4f} acc={linhas[-1]['accuracy']:.4f} "
              f"recall Fake={linhas[-1]['recall_fake']:.3f} Real={linhas[-1]['recall_real']:.3f} ({time.time() - t0:.0f}s)", flush=True)
        modelos_ajustados[nome] = mod
        if a.salvar:
            pasta = RAIZ / "modelos_salvos" / pasta_salvos
            pasta.mkdir(parents=True, exist_ok=True)
            joblib.dump(mod, pasta / f"{nome}.joblib")
    usados = list(a.modelos)
    if not a.sem_stacking and len(usados) >= 2:
        t0 = time.time()
        print("\n[stacking] gerando probabilidades fora-da-dobra no desenvolvimento...", flush=True)
        oof = np.zeros((len(yd), len(usados)))
        for k in range(5):
            tr, va = folds != k, folds == k
            for j, nome in enumerate(usados):
                m = cons[nome]().fit(Xd[tr], yd[tr])
                oof[va, j] = m.predict_proba(Xd[va])[:, 1]
        meta = LogisticRegression(C=1.0, max_iter=2000, random_state=42).fit(logit(oof), yd)
        Z = np.column_stack([logit(preds[f"p_fake_{n}"].to_numpy()) for n in usados])
        p = meta.predict_proba(Z)[:, 1]
        preds["p_fake_stacking"] = p
        linhas.append(metricas("stacking", ye, p))
        print(f"[stacking] AUC={linhas[-1]['roc_auc']:.4f} acc={linhas[-1]['accuracy']:.4f} "
              f"recall Fake={linhas[-1]['recall_fake']:.3f} Real={linhas[-1]['recall_real']:.3f} ({time.time() - t0:.0f}s)", flush=True)
        print("  pesos do meta-modelo:", {n: round(float(c), 3) for n, c in zip(usados, meta.coef_.ravel())})
        if a.salvar:
            pasta = RAIZ / "modelos_salvos" / pasta_salvos
            joblib.dump({"ordem": usados, "meta": meta}, pasta / "stacking_meta.joblib")
    tab = pd.DataFrame(linhas).set_index("modelo")
    tab.round(4).to_csv(saida / "metricas.csv")
    preds.to_csv(saida / "predicoes.csv", index=False)
    print("\n=== TESTE EXTERNO ===")
    print(tab.round(4).to_string())
    if "Dominio" in ext.columns:
        dom = ext["Dominio"].fillna("").astype(str)
        dom_fake = dom[ye == 1].value_counts()
        if len(dom_fake) > 1:
            top = dom_fake.index[0]
            mant = ~((ye == 1) & (dom == top)).to_numpy()
            print(f"\nAUC SEM o site dominante dos Fake ({top}, {int(dom_fake.iloc[0])} textos removidos); "
                  f"restam {int((ye[mant] == 1).sum())} Fake e {int((ye[mant] == 0).sum())} Real:")
            sem = {c[len('p_fake_'):]: roc_auc_score(ye[mant], preds.loc[mant, c]) for c in preds.columns if c.startswith("p_fake_")}
            print(pd.Series(sem).round(4).to_string())
            pd.Series(sem, name="roc_auc_sem_dominante").round(4).to_csv(saida / "auc_sem_site_dominante.csv")
        rec = {}
        for nome in a.modelos:
            pr = (preds[f"p_fake_{nome}"] >= 0.5).astype(int)
            g = pd.DataFrame({"dom": ext["Dominio"], "y": ye, "ok": (pr == ye)})
            rec[nome] = g.groupby(["y", "dom"])["ok"].mean()
        r = pd.DataFrame(rec).round(3)
        n = pd.DataFrame({"dom": ext["Dominio"], "y": ye}).groupby(["y", "dom"]).size().rename("n")
        r = r.join(n)
        r = r[r["n"] >= 30].sort_values(["y", "n"], ascending=[False, False])
        r.index = [("Fake" if y_ else "Real", d) for y_, d in r.index]
        print("\nAcerto por dominio (so dominios com >= 30 textos):")
        print(r.to_string())
        r.to_csv(saida / "acerto_por_dominio.csv")
    print(f"\nSaidas em {saida}")


if __name__ == "__main__":
    main()