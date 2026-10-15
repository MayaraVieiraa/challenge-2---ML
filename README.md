# Classificador de notícias falsas (FakeRecogna 2.0)

Projeto acadêmico em grupo: classificador de fake news em português treinado na FakeRecogna 2.0
(Hugging Face `recogna-nlp`, variante extrativa), com auditoria de vazamento e teste externo no Fake.br.

## Resultados
- Validação cruzada agrupada por URL (17.648 textos): stacking com ROC AUC 0,9855 e Brier 0,0436; melhor modelo individual, char n-gramas, AUC 0,9925.
- Teste externo no Fake.br (7.200 textos): SVM teve a maior AUC (0,7681); stacking obteve 0,7388. Sem o domínio Fake dominante, stacking chegou a 0,7801, em 3.865 textos.
- Teste reservado da FakeRecogna 2.0 (3.529 textos, avaliado uma única vez): stacking AUC 0,9892.
- Falsos positivos: com treino só na FakeRecogna, 26% das notícias Reais do Fake.br saem como Fake no limiar 0,5 (o Real da base de treino é quase só de 2020-2021, sobre covid; o modelo aprendeu época e tema).
- Correção: metade dos pares do Fake.br entra no treino (`--treino-externo 0.5`) e a outra metade serve de teste. Stacking: AUC 0,961; 10% dos Reais como Fake no limiar 0,5. Com três faixas no `classificar.py` (Real < 0,3, Incerto, Fake >= 0,7): 4,9% dos Reais como Fake, 81% dos Fake detectados, 14% em Incerto. Esse teste não é independente (pares das mesmas fontes); fora do diariodobrasil.org (92% dos Fake) há só 141 Fake no teste.

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
python3 modelos/avaliar_externo.py --externo dados_externos/fakebr_t100.csv --salvar                      # só mede (treino só na FakeRecogna)
python3 modelos/avaliar_externo.py --externo dados_externos/fakebr_t100.csv --treino-externo 0.5 --salvar  # modelos de produção
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