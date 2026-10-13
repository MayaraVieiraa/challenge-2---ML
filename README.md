# Classificador de notícias falsas (FakeRecogna 2.0)

Projeto acadêmico em grupo: classificador de fake news em português treinado na FakeRecogna 2.0
(Hugging Face `recogna-nlp`, variante extrativa), com auditoria de vazamento e teste externo no Fake.br.

## Resultados
- Validação cruzada agrupada por URL (17.648 textos): stacking com ROC AUC 0,9855 e Brier 0,0436; melhor modelo individual, char n-gramas, AUC 0,9925.
- Teste externo no Fake.br (7.200 textos): SVM teve a maior AUC (0,7681); stacking obteve 0,7388. Sem o domínio Fake dominante, stacking chegou a 0,7801, em 3.865 textos.
- O teste reservado da FakeRecogna 2.0 (3.529 textos) ainda não foi avaliado.

Os resultados externos são inferiores aos da validação interna: o desempenho não se transfere integralmente para notícias de outras fontes. O BERTopic fica fora do pipeline final.

## Como reproduzir
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. baixar a FakeRecogna 2.0 (extrativa) para dados/fakerecogna2_extrativa.csv
# 2. tratamento
python3 tratamento/preparar_fakerecogna2.py --variante extrativa
python3 tratamento/casar_tamanho.py --variante extrativa --largura 5
# 3. modelos, auditorias e stacking (sem BERTopic)
python3 executar_tudo.py --sem-bertopic
# 4. teste externo (clonar antes https://github.com/roneysco/Fake.br-Corpus fora do projeto)
python3 tratamento/preparar_fakebr.py --repo ~/Downloads/Fake.br-Corpus --truncar 100
python3 modelos/avaliar_externo.py --externo dados_externos/fakebr_t100.csv --salvar
```

## Estrutura
- `tratamento/`: preparo da base, deduplicação, casamento por tamanho e preparo do Fake.br
- `modelos/`: baseline, SVM, NB, char n-gramas, GBM, stacking e teste externo (veja `modelos/README.md`)
- `auditoria/`: vazamento, duplicação e vocabulário editorial
- Dados, saídas e modelos treinados não vão para o Git (ver `.gitignore`).

## Decisões importantes
- `Label 1` = Fake na 2.0. Deduplicação só por texto (por URL apagaria trechos distintos).
- Divisão com `StratifiedGroupKFold` (coluna `Grupo`); teste reservado avaliado uma única vez.
- Entrada dos modelos: só `Noticia` (títulos trazem marcadores de veredito).
- BERTopic fica fora do stacking (AUC externo abaixo dos modelos TF-IDF).