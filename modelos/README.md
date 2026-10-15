# Classificador de notícias falsas (FakeRecogna 2.0)

Projeto acadêmico em grupo: classificador de fake news em português treinado na FakeRecogna 2.0
(Hugging Face `recogna-nlp`, variante extrativa), com auditoria de vazamento e teste externo no Fake.br.

## Resultados até agora
- Validação cruzada (base casada por tamanho, 17.648 textos): AUC 0,96 a 0,98 (stacking 0,985).
- Teste externo no Fake.br (7.200 textos), modelos treinados só na FakeRecogna: AUC 0,72 a 0,77, e 26% dos Reais saem como Fake. O desempenho na base não vale para notícias de fora dela.
- Com metade dos pares do Fake.br no treino (`--treino-externo 0.5`) e teste na outra metade: stacking AUC 0,961, 10% dos Reais como Fake no limiar 0,5. Esse teste não é independente (mesmas fontes).
- Teste reservado (3.529 textos, avaliado uma única vez): stacking AUC 0,9892; char 0,9871; SVM 0,9835. Vale para a mesma coleção, não para notícias de fora.

## Classificar notícias novas
```bash
python3 modelos/avaliar_externo.py --externo dados_externos/fakebr_t100.csv --treino-externo 0.5 --salvar   # gera modelos_salvos/
python3 modelos/classificar.py --csv noticias.csv --saida resultado.csv                # coluna Noticia
```
O `p_fake` é um indício, não um veredito (em fontes e épocas novas o desempenho cai). Use textos de 50 palavras ou mais.
A classe sai em três faixas: Real (p < 0,3), Incerto (revisar) e Fake (p >= 0,7); ajuste com `--limiar-real` e `--limiar-fake`.

## Como reproduzir
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. baixar a FakeRecogna 2.0 (extrativa) para dados/fakerecogna2_extrativa.csv
# 2. tratamento
python3 tratamento/preparar_fakerecogna2.py --variante extrativa
python3 tratamento/casar_tamanho.py --variante extrativa --largura 5
# 3. modelos, auditorias e stacking
python3 executar_tudo.py --sem-bertopic
# 4. teste externo (clonar antes https://github.com/roneysco/Fake.br-Corpus fora do projeto)
python3 tratamento/preparar_fakebr.py --repo ~/Downloads/Fake.br-Corpus --truncar 100
python3 modelos/avaliar_externo.py --externo dados_externos/fakebr_t100.csv --treino-externo 0.5 --salvar
```

## Estrutura
- `tratamento/`: preparo da base, dedup, casamento por tamanho, preparo do Fake.br
- `modelos/`: baseline, SVM, NB, char n-gramas, GBM, stacking, teste externo (veja `modelos/README.md`)
- `auditoria/`: vazamento, duplicação, vocabulário editorial
- Dados, saídas e modelos treinados não vão para o Git (ver `.gitignore`).

## Decisões importantes
- `Label 1` = Fake na 2.0. Dedup só por texto (por URL apagaria trechos distintos).
- Divisão com `StratifiedGroupKFold` (coluna `Grupo`); teste reservado avaliado uma única vez.
- Entrada dos modelos: só `Noticia` (títulos trazem marcadores de veredito).
- BERTopic fica fora do stacking (AUC ~0,75).