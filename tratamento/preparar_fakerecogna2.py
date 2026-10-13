# -*- coding: utf-8 -*-
"""
tratamento/preparar_fakerecogna2.py

Trata a base FakeRecogna 2.0 (Hugging Face, grupo recogna-nlp) e a deixa no MESMO formato
das bases tratadas do projeto (colunas Rotulo, Classe, Dominio, Ano, features de texto...),
para que os scripts de modelagem funcionem sem alteracao.

O que faz:
  1. Le a base (CSV local passado em --csv, ou baixa do Hugging Face com a lib `datasets`).
  2. Descobre SOZINHO qual valor de `Label` significa Fake (a 2.0 pode usar convencao oposta
     a da 1.0, onde Classe 0 = Fake). Usa dominios de checagem (boatos.org, e-farsas...) e
     ABORTA se o resultado for ambiguo, em vez de chutar.
  3. Converte para a convencao do projeto: Classe 0 = Fake, 1 = Real; Rotulo = "Fake"/"Real".
  4. Reaproveita `preparar()` de rodar_limpeza.py (datas, dominio, features de texto).
  5. Deduplica so por Noticia repetida (mantem a 1a); cria a coluna Grupo (= URL, ou id unico se sem URL).
  6. Imprime e salva um diagnostico: balanco, tamanho do texto por classe (efeito do resumo
     das noticias reais), assinatura de lematizacao ("umar"), maiusculas, marcadores no titulo.

Uso (a partir da raiz do projeto, onde fica requirements.txt):
  pip install datasets
  python tratamento/preparar_fakerecogna2.py --variante extrativa
  python tratamento/preparar_fakerecogna2.py --variante abstrativa
  python tratamento/preparar_fakerecogna2.py --csv dados/fakerecogna2_extrativa.csv --variante extrativa

Saidas em dados_tratados/:
  FakeRecogna2_<variante>_tratada.csv
  FakeRecogna2_<variante>_dedup.csv
  FakeRecogna2_<variante>_diagnostico.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rodar_limpeza import preparar  # noqa: E402  (reaproveita a limpeza do projeto)

PASTA_DADOS = RAIZ / "dados"
PASTA_SAIDA = RAIZ / "dados_tratados"
PASTA_DADOS.mkdir(exist_ok=True)
PASTA_SAIDA.mkdir(exist_ok=True)

HF = {"extrativa": "recogna-nlp/fakerecogna2-extrativa",
      "abstrativa": "recogna-nlp/fakerecogna2-abstrativa"}

# Fontes que so publicam checagens (conteudo Fake). Usadas apenas para descobrir a convencao.
PADRAO_CHECAGEM = (r"boatos\.org|e-farsas\.com|checamos\.afp\.com|projetocomprova\.com\.br|"
                   r"aosfatos\.org|lupa\.uol|/fato-ou-fake|uol\.com\.br/confere|"
                   r"estadao-verifica")


def carregar(variante, csv):
    if csv:
        return pd.read_csv(csv)
    destino = PASTA_DADOS / f"fakerecogna2_{variante}.csv"
    if destino.exists():
        print(f"Usando arquivo ja baixado: {destino}")
        return pd.read_csv(destino)
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Instale a biblioteca:  pip install datasets   (ou passe --csv)")
    print(f"Baixando {HF[variante]} do Hugging Face ...")
    df = load_dataset(HF[variante])["train"].to_pandas()
    df.to_csv(destino, index=False, encoding="utf-8")
    print(f"Salvo em {destino}")
    return df


def descobrir_convencao(df):
    """Retorna o valor de Label que significa Fake, ou aborta se ambiguo."""
    eh_checagem = df["URL"].astype(str).str.contains(PADRAO_CHECAGEM, case=False, regex=True)
    n = int(eh_checagem.sum())
    if n < 50:
        sys.exit(f"Poucas linhas de fontes de checagem ({n}) para descobrir o rotulo. "
                 "Confira a coluna URL.")
    media = df.loc[eh_checagem, "Label"].astype(float).mean()
    outras = df.loc[~eh_checagem, "Label"].astype(float).mean()
    print(f"Fontes de checagem: {n} linhas, media de Label = {media:.3f} "
          f"(demais fontes: {outras:.3f})")
    if media >= 0.9:
        fake = 1
    elif media <= 0.1:
        fake = 0
    else:
        sys.exit("Convencao de Label ambigua (checagens nao estao quase todas em um valor). "
                 "Nao vou chutar: confira a base manualmente.")
    print(f"=> Label {fake} = Fake (convencao {'oposta' if fake == 1 else 'igual'} a da 1.0).")
    return fake


def normalizar_url(serie):
    """Transforma '', 'null', 'none', 'nan' (e nulos) em NaN. Na 2.0 varias noticias nao tem URL."""
    s = serie.astype("string").str.strip()
    return s.mask(s.str.lower().isin(["", "null", "none", "nan", "<na>"]))


def deduplicar(df):
    """Remove apenas Noticia repetida (mantem a 1a).
    NAO remove por URL: na 2.0 a mesma URL (materia de checagem) traz varios TRECHOS diferentes,
    que sao dados validos. Em vez disso, a coluna Grupo (= URL) e usada para dividir treino/teste
    sem separar trechos da mesma materia (ver modelos/_divisao.py)."""
    n0 = len(df)
    d2 = df.drop_duplicates(subset="Noticia", keep="first")
    antes = df["Rotulo"].value_counts().to_dict()
    depois = d2["Rotulo"].value_counts().to_dict()
    rep = d2[d2["URL"].notna() & d2["URL"].duplicated(keep=False)]
    info = {"linhas_antes": n0, "removidas_por_noticia": n0 - len(d2), "linhas_depois": len(d2),
            "linhas_em_url_repetida_mantidas": int(len(rep)),
            "urls_repetidas_distintas": int(rep["URL"].nunique()),
            "linhas_por_classe_antes": antes, "linhas_por_classe_depois": depois,
            "removidas_por_classe": {k: int(antes.get(k, 0) - depois.get(k, 0)) for k in antes}}
    return d2, info


def diagnostico(df):
    rel = {}
    rel["linhas"] = len(df)
    rel["balanco"] = df["Rotulo"].value_counts().to_dict()
    rel["noticia_vazia"] = int((df["Noticia"].str.strip() == "").sum())
    # tamanho do texto por classe: o artigo diz que so as noticias REAIS foram resumidas
    por_classe = df.groupby("Rotulo")["n_palavras"].describe(percentiles=[.1, .5, .9])
    rel["n_palavras_por_classe"] = {k: {c: round(float(v), 1) for c, v in r.items()}
                                    for k, r in por_classe.iterrows()}
    med = df.groupby("Rotulo")["n_palavras"].median()
    if {"Fake", "Real"} <= set(med.index) and med["Real"] > 0:
        razao = float(med["Fake"] / med["Real"])
        rel["razao_mediana_fake_real"] = round(razao, 2)
        rel["alerta_tamanho"] = not (0.67 <= razao <= 1.5)
    # a 2.0 deve estar SEM lematizacao e em minusculas
    tokens = df["Noticia"].str.split()
    rel["docs_com_umar"] = int(tokens.apply(lambda t: "umar" in t).sum())
    rel["docs_com_uma"] = int(tokens.apply(lambda t: "uma" in t).sum())
    rel["docs_com_maiuscula"] = int(df["Noticia"].str.contains(r"[A-ZÁÉÍÓÚÂÊÔÃÕÇ]").sum())
    # marcadores editoriais no titulo (vazamento de rotulo)
    t = df["Titulo"].fillna("")
    marc = t.str.contains(r"#\w+|^\s*[EÉ]\s+(?:#|fake|falso|enganoso)", case=False, regex=True)
    rel["titulo_com_marcador_por_classe_%"] = (marc.groupby(df["Rotulo"]).mean() * 100).round(1).to_dict()
    # fontes x rotulo
    rel["top_dominios"] = (df.groupby("Dominio")["Classe"].agg(["size", "mean"])
                           .sort_values("size", ascending=False).head(12)
                           .round(3).reset_index().to_dict("records"))
    return rel


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variante", choices=list(HF), default="extrativa")
    ap.add_argument("--csv", default=None, help="CSV local da base (se nao quiser baixar)")
    a = ap.parse_args()

    df = carregar(a.variante, a.csv)
    print(f"Lido: {df.shape[0]} linhas x {df.shape[1]} colunas | colunas: {list(df.columns)}")
    obrig = ["Titulo", "Subtitulo", "Noticia", "Categoria", "Data", "Autor", "URL", "Label"]
    faltando = [c for c in obrig if c not in df.columns]
    if faltando:
        sys.exit(f"Colunas ausentes: {faltando}")
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    df["URL"] = normalizar_url(df["URL"])
    df = df.dropna(subset=["Noticia", "Label"]).copy()
    # Artefato: o resumo das noticias reais deixou a palavra "none" (campo vazio) no inicio do texto
    # (146 Real x 2 Fake na base casada). Vira atalho para o modelo, entao e removida.
    df["Noticia"] = df["Noticia"].astype(str)
    n_none = int(df["Noticia"].str.contains(r"\bnone\b", case=False, regex=True).sum())
    df["Noticia"] = (df["Noticia"].str.replace(r"\bnone\b", " ", case=False, regex=True)
                     .str.replace(r"\s+", " ", regex=True).str.strip())
    df = df[df["Noticia"] != ""].copy()
    print(f"Removida a palavra 'none' de {n_none} noticias.")

    fake = descobrir_convencao(df)
    eh_fake = df["Label"].astype(int) == fake
    df["Classe"] = np.where(eh_fake, 0, 1)            # convencao do projeto: 0 = Fake, 1 = Real
    df = df.drop(columns=["Label"])

    df = preparar(df)                                  # adiciona Rotulo, Dominio, Ano, features...
    # Grupo = URL; sem URL, um id unico por linha. Usado para dividir treino/teste sem separar a mesma materia.
    df["Grupo"] = df["URL"].astype(object).where(df["URL"].notna(), ("sem_url_" + df.index.astype(str)).to_numpy())
    tratada = PASTA_SAIDA / f"FakeRecogna2_{a.variante}_tratada.csv"
    df.to_csv(tratada, index=False, encoding="utf-8-sig")

    dd, info_dedup = deduplicar(df)
    dedup = PASTA_SAIDA / f"FakeRecogna2_{a.variante}_dedup.csv"
    dd.to_csv(dedup, index=False, encoding="utf-8-sig")

    rel = diagnostico(dd)
    rel["docs_com_none_removido"] = n_none
    rel["dedup"] = info_dedup
    rel["variante"] = a.variante
    rel["label_que_significa_fake"] = int(fake)
    (PASTA_SAIDA / f"FakeRecogna2_{a.variante}_diagnostico.json").write_text(
        json.dumps(rel, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== DIAGNOSTICO (apos deduplicar) ===")
    print(json.dumps(rel, ensure_ascii=False, indent=2))
    perda = 1 - info_dedup["linhas_depois"] / max(info_dedup["linhas_antes"], 1)
    if perda > 0.05:
        print(f"\n[ALERTA] A deduplicacao removeu {perda:.1%} das linhas ({info_dedup['removidas_por_classe']}). "
              "O artigo da 2.0 reporta menos de 1% de duplicatas entre as falsas. Investigue antes de modelar.")
    if rel.get("alerta_tamanho"):
        print("\n[ALERTA] O tamanho mediano do texto difere muito entre Fake e Real "
              f"(razao {rel['razao_mediana_fake_real']}). O modelo pode aprender o tamanho. "
              "Considere truncar ambos ao mesmo numero de palavras.")
    if rel["docs_com_umar"] > 0.01 * rel["linhas"]:
        print("\n[ALERTA] Ha muitos 'umar': a base parece AINDA lematizada. Confira a variante.")
    print(f"\nSalvo: {tratada.name}, {dedup.name}")


if __name__ == "__main__":
    main()