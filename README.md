# Desafio Participa DF - README Operacional (Didatico)

Este README explica, passo a passo, como configurar o ambiente e executar o classificador de textos que identifica pedidos com dados pessoais. A ideia e evitar a publicacao indevida de informacoes sensiveis.

## O que esta solucao faz
- Lê textos de pedidos de acesso a informacao
- Detecta dados pessoais com regras (regex) e com um modelo BERT treinado com dados similares à solicitações de acesso a informação.
- Classifica cada pedido como publico ou privado

## Pre-requisitos
Antes de iniciar, verifique se voce tem:
- Python 3.12+ instalado
- pip
- Git (se precisar clonar o repositorio)
- Sistema operacional Windows, macOS ou Linux

## Como preparar o ambiente
Siga os passos na ordem.

1) (Opcional) Clonar o repositorio
```bash
git clone https://github.com/lyscoutinho/hackathon-participaDf.git
cd hackathon-participaDf
```

2) Criar o ambiente virtual
```bash
python -m venv venv
```

3) Ativar o ambiente virtual
- Windows (PowerShell)
```powershell
venv\Scripts\Activate.ps1
```
- Linux/macOS
```bash
source venv/bin/activate
```

4) Atualizar o pip
```bash
python -m pip install --upgrade pip
```

5) Instalar as dependencias
```bash
pip install -r requirements.txt
```

## Como executar o classificador (classificaDados.py)
O script tem dois modos: manual e batch (planilha).

### 1) Modo manual (interativo)
Use quando quiser testar frases rapidamente.
```bash
python src/classificaDados.py
```
O terminal vai pedir um texto. Digite e pressione Enter.

### 2) Modo batch com Excel (.xlsx)
Use quando quiser classificar varias linhas de uma planilha.
```bash

python .\src\classificaDados.py `
>>   --xlsx "caminhoEntrada\arquivoEntrada.xlsx" `
>>   --output "caminhoSaida\arquivoSaida.xlsx" `
>>   --text-column "NomeDaColunaDeTexto" `
>>   --max-length-chunk 384 `
>>   --overlap 128 `
>>   --batch-size 32

```

#### Argumentos principais
- `--xlsx`: caminho do arquivo de entrada (.xlsx)
- `--output`: caminho do arquivo de saida (.xlsx)
- `--text-column`: nome da coluna com o texto
- `--batch-size`: tamanho do lote (padrao: 32)
- `--max-length-chunk`: maximo de tokens por janela (padrao: 384)
- `--overlap`: sobreposicao entre janelas (padrao: 128)
- `--max-chunks`: limite opcional de janelas

## Formato de entrada
### Entrada no modo batch
- Arquivo Excel `.xlsx`
- Uma coluna com o texto do pedido (indique o nome da coluna de texto. Exemplo: `Texto Mascarado`)
- Cada linha representa um pedido

Exemplo (colunas minimas):
- `Texto Mascarado`

## Formato de saida
### Saida no modo batch
Arquivo Excel com as mesmas linhas da entrada e novas colunas adicionais com os respectivos nomes:
- `pred_label`: 0 (publico) ou 1 (privado)
- `pred_text`: "publico" ou "privado"
- `score_privado` e `score_publico`: quanto o modelo classificou o dado como publico ou privado
- `best_chunk_index` e `best_chunk_score_privado` o índice do melhor chunk identificado pelo modelo eo score dele
- `best_chunk_snippet`: trecho mais relevante
- `num_chunks`: numero de chunks identificados
- `decision_reason`: motivo da decisao

## Objetos e arquivos importantes
- `src/classificaDados.py`: script de inferencia (manual e por arquivo)
- `models/bert_finetuned/`: modelo treinado usado na classificacao
- `requirements.txt`: lista de requisitos do projeto


## Dicas rapidas
- Se tiver GPU, o PyTorch usa automaticamente.
- Se sua planilha tiver outra coluna, use `--text-column`.
