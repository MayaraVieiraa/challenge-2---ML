# -*- coding: utf-8 -*-
"""
_divisao.py: divisao dev/teste e folds POR GRUPO (unico codigo de split do projeto).

Por que por grupo: na FakeRecogna 2.0 uma mesma materia de checagem (mesma URL) aparece em varias
linhas com trechos diferentes. Se trechos da mesma materia caem no treino e no teste, o modelo
"reconhece a materia" e a nota fica inflada. Aqui, todas as linhas com a mesma URL ficam do mesmo
lado. Linhas sem URL formam grupos de uma linha so.

Protocolo (igual ao anterior, trocando so o tipo de fold):
  teste reservado ~20% = 1 fold de StratifiedGroupKFold(5, shuffle, seed 42)
  validacao cruzada    = StratifiedGroupKFold(5, shuffle, seed 42) sobre os ~80% restantes
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42


def grupos_de(caminho_csv, n):
    """Le a coluna Grupo do CSV; sem ela, cada linha e seu proprio grupo (aviso)."""
    try:
        g = pd.read_csv(caminho_csv, usecols=["Grupo"])["Grupo"].astype(str).to_numpy()
        if len(g) == n:
            return g
    except (ValueError, KeyError):
        pass
    print("[aviso] coluna 'Grupo' ausente: usando divisao SEM grupos (pode inflar a nota).", flush=True)
    return np.arange(n).astype(str)


def dividir_grupos(y, grupos, seed=SEED):
    ids = np.arange(len(y))
    sg = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    ids_dev, ids_te = next(sg.split(ids, y, groups=grupos))
    ids_dev, ids_te = np.sort(ids_dev), np.sort(ids_te)
    folds = np.full(len(ids_dev), -1)
    sg2 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for k, (_, va) in enumerate(sg2.split(ids_dev, y[ids_dev], groups=grupos[ids_dev])):
        folds[va] = k
    assert not (set(grupos[ids_dev]) & set(grupos[ids_te])), "grupo em dev e teste"
    return ids_dev, ids_te, folds