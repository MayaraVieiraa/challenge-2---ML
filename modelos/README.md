# Modelos — Protocolo Comum

Todos os scripts desta pasta seguem o mesmo protocolo, para que os resultados
sejam comparáveis entre si.

## Como instalar e executar

Não é obrigatório criar uma pasta `venv`. Com o Python 3 instalado, abra o
PowerShell na pasta raiz do projeto (a pasta que contém `requirements.txt`) e
instale as dependências no ambiente Python disponível:

```powershell
python --version
python -m pip install -r requirements.txt
```

Se o comando `python` não estiver disponível no Windows, use `py` no lugar:

```powershell
py --version
py -m pip install -r requirements.txt
```

Para baixar e preparar a base FakeRecogna2 Extrativa:

```powershell
python tratamento/preparar_fakerecogna2.py
```

Para executar a análise exploratória:

```powershell
python eda/eda.py
```

Os resultados são gravados em `dados_tratados/` e `eda_outputs/`. Os comandos
devem ser executados a partir da raiz do projeto. Instalar as dependências sem
`venv` pode afetar outros projetos que usem o mesmo Python.

Para rodar TODA a modelagem (modelos, análise de erros, auditorias e stacking) de uma vez:

```powershell
python executar_tudo.py                      # padrão: só Noticia (sem o vazamento do título)
python executar_tudo.py --entrada titulo_noticia  # protocolo ORIGINAL com título (saídas em *_titulo_noticia/)
python executar_tudo.py --backend tfidf-svd  # BERTopic offline, sem baixar modelo
python executar_tudo.py --final              # + avaliação única no teste (só no fim)
```

Ou script a script: `python modelos/treinar_baseline.py`, `treinar_svm.py`, `treinar_nb.py`,
`treinar_bertopic.py`, `analisar_erros.py`, `auditoria/auditar_*.py` e, por último,
`modelos/stacking.py`.

## Base de dados

Todos os modelos e auditorias leem de:

  dados_tratados/FakeRecogna2_extrativa_dedup.csv

- Entrada padrão: somente `Noticia`. As opções `titulo_noticia` e `titulo_limpo_noticia` continuam disponíveis, mas conclusões de auditorias da base anterior não devem ser generalizadas para esta base sem reexecução.
- Features proibidas: `Categoria`, `Dominio`, `URL`, `Autor`, `Classe`,
  `Rotulo`, `Data`, `Ano`, `AnoMes`, e as 10 features estruturais
  (n_caracteres, n_palavras, etc.)

## Vetorização

    TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.9,
        sublinear_tf=True,
        strip_accents="unicode",
    )

Mesmos parâmetros para os três modelos.

## Split

`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`

- Validação cruzada de 5 folds, estratificada por classe
- Cada modelo reporta média e desvio-padrão das métricas entre folds
- `random_state=42` fixo para reprodutibilidade

O conjunto de teste final (80/20) é reservado para a entrega final, não
usado durante a comparação.

## Métricas reportadas

Todas as métricas a seguir devem ser salvas em `metricas.json`:

- `accuracy` — referência apenas, não usada para escolha
- `roc_auc` — métrica principal de discriminação
- `pr_auc` — métrica complementar
- `brier` — métrica principal de calibração
- `precision`, `recall`, `f1` — por classe (Fake/Real)
- `roc_auc_ic95` — intervalo de confiança via bootstrap (1000 reamostragens)

## Calibração

- **Logistic Regression**: medir calibração nativa, sem `CalibratedClassifierCV`
- **Linear SVM**: obrigatório usar `CalibratedClassifierCV`; testar
  `method="sigmoid"` e `method="isotonic"` e reportar ambos
- **Multinomial NB**: obrigatório usar `CalibratedClassifierCV`; reportar
  Brier antes e depois da calibração

## Saída

Cada script salva em `modelos_outputs/<nome_do_modelo>/`:

    metricas.json
    reliability.png
    matriz_confusao.png
    classification_report.txt
    coeficientes.csv        (apenas o baseline)

## Critério de comparação entre modelos

A ordem de decisão para escolher o modelo vencedor é:

1. Maior `roc_auc` com IC95% **não sobreposto** ao baseline
2. Menor `brier`
3. Menor complexidade em caso de empate

Acurácia **não é critério de escolha**.

## Scripts desta pasta

| Script | Responsável | O que faz |
|---|---|---|
| `treinar_baseline.py` | Mayara | TF-IDF + Regressão Logística (calibração nativa), coeficientes |
| `treinar_svm.py` | Rayssa | LinearSVC + CalibratedClassifierCV (sigmoid e isotonic) |
| `treinar_nb.py` | Rafaella | MultinomialNB + CalibratedClassifierCV (Brier antes/depois) |
| `treinar_extras.py` | Mayara | char n-gramas + SVM calibrado; TF-IDF→SVD→GBM (diversidade para o stacking) |
| `treinar_bertopic.py` | Mayara | BERTopic (tópicos) + LR; tabela tópico x rótulo |
| `analisar_erros.py` | Mayara | 50 exemplos por quadrante para codificação manual |
| `stacking.py` | Equipe | tabela comparativa, soft voting, stacking e decisão do protocolo |
| `_entrada.py` | — | único código compartilhado: monta o texto de entrada |

Todos gravam `oof_<modelo>.csv` (probabilidades fora-do-fold) e `teste_<modelo>.csv`
(probabilidades no teste, SEM calcular métrica de teste). O split é idêntico em todos:
ids = posição da linha no CSV; teste 80/20 estratificado (seed 42) + 5 folds no restante.

## Parâmetro `--entrada` (adicionado após a auditoria de vazamento)

`noticia` (padrão), `titulo_noticia` ou `titulo_limpo_noticia`. Execute novamente as auditorias
para avaliar vazamento editorial na base atual antes de interpretar resultados com título.

## Auditorias (`auditoria/`)

`auditar_vazamento.py`, `auditar_duplicacao.py` e `auditar_vocabulario_editorial.py` escrevem em
`auditoria_outputs/` (`RESUMO_AUDITORIA.md` consolida). Usam só o desenvolvimento e os mesmos folds.
