# -*- coding: utf-8 -*-
"""
_entrada.py — monta o TEXTO de entrada dos modelos (único código compartilhado entre os scripts).

Os scripts de modelo continuam autocontidos para split, vetorização e métricas (protocolo
comum). Só a escolha do texto de entrada foi centralizada aqui porque ela passou a ser um
parâmetro experimental depois da auditoria de vazamento (ver auditoria/ e o relatório):

  titulo_noticia        PROTOCOLO OFICIAL: Titulo + " " + Noticia.
  noticia               só a coluna Noticia (como no script original da Rayssa).
  titulo_limpo_noticia  Titulo sem as marcas editoriais óbvias + Noticia. Remove:
                          - prefixos de veredito  ("É #FAKE ...", "É falso ...", "É enganoso ...")
                          - qualquer hashtag      ("... #boato")
                        A auditoria mostrou que "#boato" aparece em ~43% dos títulos Fake e
                        "É #FAKE" em outros ~17%, quase nunca em Real: é um rótulo escrito.

Nenhuma outra coluna (Categoria, Dominio, URL, Autor, Subtitulo, Data, features
estruturais...) é lida pelos modelos.
"""
import re

import numpy as np
import pandas as pd

ENTRADAS = ("titulo_noticia", "noticia", "titulo_limpo_noticia")

_PREFIXO_VEREDITO = re.compile(
    r"^\s*[EÉée]\s+(?:#\s*)?(?:fake|fato|boato|falso|falsa|enganoso|enganosa|verdade|verdadeiro|"
    r"exagerado|impreciso|sem\s+provas|s[áa]tira)\b[:,.!\-–—]*\s*", re.IGNORECASE)
_HASHTAG = re.compile(r"#\w+")


def limpar_titulo(titulo: str) -> str:
    titulo = _PREFIXO_VEREDITO.sub("", titulo)
    titulo = _HASHTAG.sub(" ", titulo)
    return " ".join(titulo.split())


def montar_texto(df: pd.DataFrame, entrada: str = "titulo_noticia") -> np.ndarray:
    if entrada not in ENTRADAS:
        raise ValueError(f"entrada inválida: {entrada!r}; use uma de {ENTRADAS}")
    titulo = df["Titulo"].fillna("").astype(str)
    noticia = df["Noticia"].fillna("").astype(str)
    if entrada == "noticia":
        texto = noticia
    elif entrada == "titulo_limpo_noticia":
        texto = titulo.map(limpar_titulo) + " " + noticia
    else:
        texto = titulo + " " + noticia
    return texto.str.strip().to_numpy()
