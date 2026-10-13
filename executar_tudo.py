# -*- coding: utf-8 -*-
"""
executar_tudo.py — roda o pipeline de modelagem inteiro, na ordem certa.

    1. modelos base : baseline (LR) -> SVM -> NB -> BERTopic
    2. análise de erros do baseline (amostra de 50 por quadrante)
    3. auditorias   : vazamento, duplicação, vocabulário editorial  (+ RESUMO_AUDITORIA.md)
    4. stacking     : comparação, soft voting, stacking, decisão pelo critério do protocolo

Exemplos (a partir da raiz do projeto):
    python executar_tudo.py --base FakeRecogna2_extrativa_dedup.csv    # SEM casar por tamanho (para comparar; nota inflada)
    python executar_tudo.py --base FakeRecogna2_abstrativa_casada.csv  # outra variante da base
    python executar_tudo.py                          # padrão: só Noticia (sem vazamento do título)
    python executar_tudo.py --entrada titulo_noticia # protocolo ORIGINAL com título (saídas em modelos_outputs_titulo_noticia/)
    python executar_tudo.py --backend tfidf-svd      # BERTopic offline (sem baixar modelo do Hugging Face)
    python executar_tudo.py --sem-bertopic           # pula o BERTopic
    python executar_tudo.py --final                  # + avaliação ÚNICA no teste reservado (só no fim!)
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
PY = sys.executable


def rodar(nome, args):
    print(f"\n{'=' * 78}\n>> {nome}\n{'=' * 78}", flush=True)
    t0 = time.time()
    r = subprocess.run([PY] + args, cwd=RAIZ)
    print(f"-- {nome}: {'ok' if r.returncode == 0 else 'FALHOU (código %d)' % r.returncode} em {time.time() - t0:.0f}s", flush=True)
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="FakeRecogna2_extrativa_casada.csv",
                    help="arquivo em dados_tratados/ (padrao: a base CASADA por tamanho; _dedup = sem casar)")
    ap.add_argument("--entrada", default="noticia", choices=["titulo_noticia", "noticia", "titulo_limpo_noticia"])
    ap.add_argument("--backend", default="sbert", choices=["sbert", "tfidf-svd"], help="embeddings do BERTopic")
    ap.add_argument("--sem-bertopic", action="store_true")
    ap.add_argument("--sem-auditoria", action="store_true")
    ap.add_argument("--rapido", action="store_true", help="auditorias em modo rápido")
    ap.add_argument("--final", action="store_true", help="avalia também o teste reservado (uma única vez)")
    a = ap.parse_args()

    os.environ["BASE_DADOS"] = a.base                      # lido por todos os scripts (filhos herdam)
    if not (RAIZ / "dados_tratados" / a.base).exists():
        sys.exit(f"Nao achei dados_tratados/{a.base}. Rode antes: python tratamento/preparar_fakerecogna2.py e python tratamento/casar_tamanho.py")
    tag = Path(a.base).stem.replace("FakeRecogna2_", "").replace("_dedup", "_semcasar")
    sufixo = f"_{tag}" + ("" if a.entrada == "noticia" else f"_{a.entrada}")
    base = f"modelos_outputs{sufixo}"
    aud = f"auditoria_outputs{sufixo}"
    comum = ["--entrada", a.entrada, "--saida-base", base]
    falhas = []

    passos = [("Baseline (LR)", ["modelos/treinar_baseline.py"] + comum),
              ("SVM calibrado (Rayssa)", ["modelos/treinar_svm.py"] + comum),
              ("Naive Bayes calibrado (Rafaella)", ["modelos/treinar_nb.py"] + comum)]
    passos.append(("Modelos extras (char n-gramas, SVD+GBM)", ["modelos/treinar_extras.py"] + comum))
    if not a.sem_bertopic:
        passos.append(("BERTopic", ["modelos/treinar_bertopic.py", "--backend", a.backend] + comum))
    for nome, args in passos:
        if not rodar(nome, args):
            falhas.append(nome)

    if not rodar("Análise de erros do baseline", ["modelos/analisar_erros.py", "--entrada", a.entrada, "--saida-base", base]):
        falhas.append("Análise de erros")

    if not a.sem_auditoria:
        extra = ["--rapido"] if a.rapido else []
        for nome, scr, ex in [("Auditoria de vazamento", "auditar_vazamento.py", extra),
                              ("Auditoria de duplicação", "auditar_duplicacao.py", ["--entrada", a.entrada]),
                              ("Auditoria de vocabulário editorial", "auditar_vocabulario_editorial.py", extra)]:
            if not rodar(nome, [f"auditoria/{scr}", "--saida-base", aud] + ex):
                falhas.append(nome)
        partes = []
        for n in ("vazamento", "duplicacao", "vocabulario_editorial"):
            f = RAIZ / aud / f"{n}.md"
            if f.exists():
                partes.append(f.read_text(encoding="utf-8"))
        (RAIZ / aud / "RESUMO_AUDITORIA.md").write_text("# Resumo das auditorias\n\n" + "\n\n".join(partes), encoding="utf-8")

    mods = ["baseline", "svm", "nb", "char", "gbm"] + ([] if a.sem_bertopic else ["bertopic"])
    st = ["modelos/stacking.py", "--saida-base", base, "--modelos"] + mods
    if a.final:
        st.append("--avaliar-teste")
    if not rodar("Stacking e decisão", st):
        falhas.append("Stacking")

    print("\n" + "=" * 78)
    print("CONCLUÍDO" if not falhas else f"CONCLUÍDO COM FALHAS em: {', '.join(falhas)}")
    print(f"Modelos: {RAIZ / base}\nAuditorias: {RAIZ / aud}")
    sys.exit(1 if falhas else 0)


if __name__ == "__main__":
    main()