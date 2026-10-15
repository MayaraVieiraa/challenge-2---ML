# -*- coding: utf-8 -*-
"""
Funções comuns às auditorias (vazamento, duplicação, vocabulário editorial).

IMPORTANTE — as auditorias NÃO tocam no conjunto de teste final para avaliar desempenho:
os testes baseados em modelo usam só os 80% de desenvolvimento e os MESMOS 5 folds do
protocolo. (A auditoria de duplicação olha os TEXTOS do teste para checar sobreposição com
o desenvolvimento — isso não usa rótulos nem mede desempenho.)
As colunas proibidas (Categoria, Dominio, Autor, Ano...) são lidas aqui apenas como OBJETO
de auditoria, nunca como feature dos modelos.
"""
import os
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "modelos"))
from _divisao import dividir_grupos, grupos_de  # noqa: E402
from _entrada import ENTRADAS, limpar_titulo, montar_texto  # noqa: E402,F401

SEED = 42
DADOS = RAIZ / "dados_tratados" / os.environ.get("BASE_DADOS", "FakeRecogna2_extrativa_casada.csv")


def pasta_saida(base=None):
    p = Path(base) if base else RAIZ / "auditoria_outputs"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ------------------------------------------------------------------ dados / split (igual aos modelos)
def carregar_df():
    df = pd.read_csv(DADOS)
    df["id"] = np.arange(len(df))
    df["y"] = (df["Rotulo"].astype(str).str.strip().str.lower() == "fake").astype(int)
    return df


def dividir(y):
    """Mesmo split dos modelos (modelos/_divisao.py): agrupado por URL."""
    return dividir_grupos(y, grupos_de(DADOS, len(y)))


def tfidf_protocolo(**kw):
    p = dict(ngram_range=(1, 2), min_df=5, max_df=0.9, sublinear_tf=True, strip_accents="unicode")
    p.update(kw)
    return TfidfVectorizer(**p)


def pipeline_lr(C=1.0, **kw):
    return Pipeline([("tfidf", tfidf_protocolo(**kw)),
                     ("lr", LogisticRegression(C=C, max_iter=2000, random_state=SEED))])


# ------------------------------------------------------------------ CV rápido (mesmos folds)
def _fold(k, X, y, folds, fabrica):
    tr, va = folds != k, folds == k
    m = fabrica().fit(X[tr], y[tr])
    return k, m.predict_proba(X[va])[:, 1]


def cv_oof(X, y, folds, fabrica=pipeline_lr, n_jobs=2):
    """Probabilidades fora-do-fold com os 5 folds do protocolo."""
    X = X if hasattr(X, "tocsr") else np.asarray(X)
    res = Parallel(n_jobs=n_jobs)(delayed(_fold)(k, X, y, folds, fabrica) for k in range(5))
    oof = np.zeros(len(y))
    for k, p in res:
        oof[folds == k] = p
    return oof


def resumo_metricas(y, p):
    return {"roc_auc": float(roc_auc_score(y, p)), "brier": float(brier_score_loss(y, p)),
            "accuracy": float(accuracy_score(y, p >= 0.5))}


def auc_ic95(y, p, n=500, seed=SEED):
    rng = np.random.default_rng(seed)
    a = []
    while len(a) < n:
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            a.append(roc_auc_score(y[i], p[i]))
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


# ------------------------------------------------------------------ texto
def normalizar(texto: str) -> str:
    """minúsculas, sem acentos, sem pontuação, espaços únicos (para comparar textos)."""
    t = unicodedata.normalize("NFKD", str(texto).lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^\w\s]", " ", t).split())


# ------------------------------------------------------------------ léxico editorial
# Padrões aplicados a texto normalizado (minúsculo, sem acento). Começam em fronteira de palavra
# e usam radicais, porque o corpo do texto está lematizado ("checar", "circular", "desmentir").
LEXICO_EDITORIAL = {
    # o texto SENTENCIA a notícia: palavras de veredito/refutação típicas de checagem
    "veredito": r"\b(boat\w*|fake\w*|fals\w*|engan\w*|ment(?:ir|ira|iras|iroso\w*)\b|desment\w*|"
                r"fraud\w*|verdadeir\w*|inverdade\w*|verdade|hoax|golpe\w*|sem\s+provas|nao\s+ha\s+evidenc\w*|"
                r"descontextualiz\w*|distorc\w*|manipul\w*|satira\w*|montagem|adulterad\w*)",
    # o texto descreve o ato de checar / cita agências e sites de checagem
    "checagem": r"\b(chec\w*|verific\w*|apurac\w*|apurar|apuramos|comprova\w*|fact\s*check\w*|fato\s+ou\s+fake|"
                r"e-?farsas|boatos\s*org|aos\s+fatos|lupa|afp|agencia\s+lupa|esclarec\w*|desminti\w*)",
    # o texto descreve a circulação viral do conteúdo (típico de matéria SOBRE um boato)
    "circulacao": r"\b(circul\w*|compartilh\w*|viraliz\w*|corrente|whatsapp|facebook|instagram|telegram|tiktok|"
                  r"redes?\s+sociais?|rede\s+social|postagem|postag\w*|mensagem|internauta\w*|usuario\w*)",
}
_LEX_RE = {k: re.compile(v) for k, v in LEXICO_EDITORIAL.items()}


def contar_marcadores(textos):
    """DataFrame com a contagem de ocorrências do léxico editorial por documento."""
    norm = [normalizar(t) for t in textos]
    out = {k: [len(r.findall(t)) for t in norm] for k, r in _LEX_RE.items()}
    df = pd.DataFrame(out)
    df["total"] = df.sum(axis=1)
    return df


def categoria_editorial(termo: str):
    """Categoria do léxico editorial a que um termo (uni/bigrama) pertence, ou ''."""
    t = normalizar(termo)
    for k, r in _LEX_RE.items():
        if r.search(t):
            return k
    return ""


def remover_lexico(texto: str, categorias=("veredito", "checagem", "circulacao")) -> str:
    t = normalizar(texto)
    for k in categorias:
        t = _LEX_RE[k].sub(" ", t)
    return " ".join(t.split())


# ------------------------------------------------------------------ saída
def veredito(nivel: str, texto: str) -> str:
    icone = {"CRITICO": "🔴", "ATENCAO": "🟠", "OK": "🟢", "INFO": "🔵"}[nivel]
    return f"{icone} **{nivel}** — {texto}"


def salvar_md(pasta: Path, nome: str, titulo: str, linhas):
    (pasta / f"{nome}.md").write_text(f"## {titulo}\n\n" + "\n".join(linhas) + "\n", encoding="utf-8")


def salvar_json(pasta: Path, nome: str, obj):
    (pasta / f"{nome}.json").write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float),
                                        encoding="utf-8")


def tabela_md(df: pd.DataFrame, floatfmt="{:.4f}") -> str:
    cols = list(df.columns)
    linhas = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cel = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r.tolist()]
        linhas.append("| " + " | ".join(cel) + " |")
    return "\n".join(linhas)
