---
name: tavily-research
description: Pesquisa informações atuais na web usando a API do Tavily. Use quando o usuário pedir pesquisas, notícias recentes, dados que podem ter mudado, ou fontes para verificar uma afirmação.
---

# Pesquisa com Tavily

## Quando usar
- Perguntas sobre fatos recentes, preços, cargos atuais ou eventos em andamento
- Quando o usuário pedir fontes ou links
- Quando a resposta depender de informação que pode estar desatualizada

## Como usar
1. Reformule a pergunta do usuário em 3 a 6 palavras-chave.
2. Execute: `python scripts/search.py "consulta"`
3. Leia os campos `answer` e `results` (cada um tem `title`, `url`, `content` e `raw_content`). O `raw_content` traz o texto completo da página; use-o quando precisar de detalhes que o `content` não mostra.
4. Sintetize a resposta em suas palavras e cite as fontes com links.

## Regras
- Não copie trechos longos dos resultados; resuma.
- Se os resultados forem contraditórios, faça uma nova busca com termos mais específicos.
- Se nada relevante aparecer, diga isso ao usuário em vez de inventar.