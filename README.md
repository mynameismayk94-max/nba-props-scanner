# NBA Props Scanner

Painel de análise pré-jogo de estatísticas de jogadores da NBA.

## Mercados monitorados

- Pontos
- Assistências
- Rebotes
- Bolas de 3 pontos convertidas
- Tentativas de arremessos

## Objetivo

Reduzir o tempo de análise dos jogos, identificando jogadores que merecem investigação com base em médias recentes, minutos em quadra e contexto do adversário.

## Tecnologias

- Python
- FastAPI
- nba_api

## Executar a API

Instale as dependências:

    pip install -r requirements.txt

Inicie o servidor:

    uvicorn app.main:app --host 0.0.0.0 --port 8000

Acesse a documentação da API em:

    http://localhost:8000/docs

## Importante

O scanner é uma ferramenta de triagem estatística, não uma garantia de resultado. Os candidatos precisam ser verificados considerando lesões, escalações, minutos esperados e linhas disponíveis na casa de apostas.
