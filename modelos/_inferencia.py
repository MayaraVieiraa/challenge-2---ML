# -*- coding: utf-8 -*-
"""
_inferencia.py: tratamento do TEXTO de uma noticia nova, igual ao da base de treino.

A base 2.0 vem em minusculas, sem lematizacao, e o conjunto de modelagem (casado por tamanho) so tem
textos de ~50 a 190 palavras. Por isso, antes de classificar qualquer noticia de fora, aplica-se aqui
o mesmo tratamento: minusculas, remocao do artefato "none", espacos normalizados e corte nas
primeiras `truncar` palavras (o mesmo corte vale para Fake e Real).
"""
import re

_NONE = re.compile(r"\bnone\b", re.IGNORECASE)
_ESPACOS = re.compile(r"\s+")


def limpar_noticia(texto, truncar=None):
    t = str(texto).replace("﻿", " ").lower()
    t = _NONE.sub(" ", t)
    t = _ESPACOS.sub(" ", t).strip()
    if truncar:
        t = " ".join(t.split()[:truncar])
    return t