# -*- coding: utf-8 -*-
"""
tratamento/rodar_limpeza.py

Limpa e prepara as bases FakeRecogna para modelagem.
NÃO gera gráficos — apenas limpeza, engenharia de atributos e exportação.

Entrada : dados/FakeRecogna.xlsx
          dados/FakeRecogna_no_removal_words.xlsx
Saída   : dados_tratados/FakeRecogna_tratada.csv
          dados_tratados/NoRemovalWords_tratada.csv

Uso:
    python tratamento/rodar_limpeza.py
"""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# CAMINHOS (resolvidos a partir da raiz do projeto, não do cwd)
# =============================================================================
RAIZ = Path(__file__).resolve().parent.parent
PASTA_DADOS = RAIZ / "dados"
PASTA_SAIDA = RAIZ / "dados_tratados"
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

ARQUIVOS = {
    "FakeRecogna":    "FakeRecogna.xlsx",
    "NoRemovalWords": "FakeRecogna_no_removal_words.xlsx",
}

COLS_OBRIGATORIAS = [
    "Titulo", "Subtitulo", "Noticia", "Categoria",
    "Data", "Autor", "URL", "Classe",
]

MAPA_CLASSE = {0: "Fake", 1: "Real"}

# =============================================================================
# HELPERS
# =============================================================================
RE_DATA = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})")


def extrair_data(valor):
    """Extrai a primeira dd/mm/aaaa de formatos variados."""
    if valor is None or (
        not isinstance(valor, (list, tuple, dict)) and pd.isna(valor)
    ):
        return pd.NaT
    if isinstance(valor, (pd.Timestamp, np.datetime64)) or hasattr(valor, "year"):
        return pd.Timestamp(valor)
    m = RE_DATA.search(str(valor))
    if not m:
        return pd.NaT
    d, mth, y = m.groups()
    return pd.to_datetime(
        f"{int(d):02d}/{int(mth):02d}/{y}",
        format="%d/%m/%Y",
        errors="coerce",
    )


def limpar_texto(s):
    """Remove espaços repetidos e faz strip."""
    return s.astype("string").str.replace(r"\s+", " ", regex=True).str.strip()


def _tam_medio_palavra(t):
    if not isinstance(t, list) or len(t) == 0:
        return 0.0
    return float(np.mean([len(w) for w in t]))


def _n_palavras_unicas(t):
    if not isinstance(t, list):
        return 0
    return len(set(t))


# =============================================================================
# PIPELINE DE LIMPEZA
# =============================================================================
def preparar(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica limpeza e cria atributos derivados. Retorna novo DataFrame."""
    df = df.copy()

    # -- linhas inúteis --------------------------------------------------------
    df = df.dropna(how="all")
    df = df.dropna(subset=["Noticia", "Classe"])

    # -- normalização de texto -------------------------------------------------
    for col in ["Titulo", "Subtitulo", "Autor", "Categoria"]:
        df[col] = limpar_texto(df[col])
    df["Noticia"] = df["Noticia"].astype(str)

    # -- rótulo ----------------------------------------------------------------
    df["Classe"] = df["Classe"].astype(int)
    df["Rotulo"] = df["Classe"].map(MAPA_CLASSE)

    # -- domínio a partir da URL ----------------------------------------------
    df["Dominio"] = (
        df["URL"].astype(str)
        .str.extract(r"https?://([^/]+)")[0]
        .str.replace(r"^www\.", "", regex=True)
    )

    # -- datas -----------------------------------------------------------------
    df["Data_dt"] = df["Data"].apply(extrair_data)
    df["Ano"] = df["Data_dt"].dt.year
    df["AnoMes"] = df["Data_dt"].dt.to_period("M").astype("string")

    # -- features de texto -----------------------------------------------------
    tokens = df["Noticia"].str.split()
    df["n_caracteres"] = df["Noticia"].str.len()
    df["n_palavras"] = tokens.str.len()
    df["n_palavras_unicas"] = tokens.apply(_n_palavras_unicas)
    df["diversidade_lexica"] = (
        df["n_palavras_unicas"] / df["n_palavras"].replace(0, np.nan)
    )
    df["tam_medio_palavra"] = tokens.apply(_tam_medio_palavra)

    # -- features do título/subtítulo -----------------------------------------
    df["n_palavras_titulo"] = df["Titulo"].fillna("").str.split().str.len()
    df["tem_subtitulo"] = df["Subtitulo"].notna().astype(int)
    df["titulo_maiusculas_%"] = (
        df["Titulo"].fillna("").apply(
            lambda t: 100 * sum(c.isupper() for c in t)
            / max(sum(c.isalpha() for c in t), 1)
        )
    )
    df["n_exclamacoes_titulo"] = df["Titulo"].fillna("").str.count("!")
    df["n_interrogacoes_titulo"] = df["Titulo"].fillna("").str.count(r"\?")

    return df


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 80)
    print("LIMPEZA E PREPARAÇÃO DAS BASES FAKERECOGNA")
    print("=" * 80)
    print(f"Raiz do projeto : {RAIZ}")
    print(f"Dados           : {PASTA_DADOS}")
    print(f"Saída           : {PASTA_SAIDA}")
    print()

    for nome, arquivo in ARQUIVOS.items():
        caminho = PASTA_DADOS / arquivo
        if not caminho.exists():
            raise FileNotFoundError(f"Não achei: {caminho}")

        print("-" * 80)
        print(f"[{nome}] Lendo {caminho.name} ...")
        df_raw = pd.read_excel(caminho)

        # valida colunas
        faltando = [c for c in COLS_OBRIGATORIAS if c not in df_raw.columns]
        if faltando:
            raise KeyError(
                f"A base '{nome}' não tem as colunas: {faltando}. "
                f"Colunas encontradas: {list(df_raw.columns)}"
            )

        print(f"          {df_raw.shape[0]} linhas x {df_raw.shape[1]} colunas")

        # limpeza
        df = preparar(df_raw)
        removidas = len(df_raw) - len(df)
        datas_ok = df["Data_dt"].notna().mean() * 100

        print(f"          {len(df)} linhas após limpeza "
              f"(removidas: {removidas})")
        print(f"          datas reconhecidas: {datas_ok:.1f}%")
        print(f"          distribuição Rotulo: "
              f"{df['Rotulo'].value_counts().to_dict()}")

        # salva
        destino = PASTA_SAIDA / f"{nome}_tratada.csv"
        df.to_csv(destino, index=False, encoding="utf-8-sig")
        print(f"          -> salvo em {destino.relative_to(RAIZ)}")
        print()

    print("=" * 80)
    print("CONCLUÍDO.")
    print(f"Arquivos tratados em: {PASTA_SAIDA.relative_to(RAIZ)}/")
    print("=" * 80)


if __name__ == "__main__":
    main()