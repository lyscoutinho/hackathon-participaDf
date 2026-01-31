# Desafio Participa DF - Acesso a informacao
### Lider: Cayo Alencar; Grupo: Cayo Alencar, Filipe Bressanelli e Lys Coutinho.

Este documento resume o desenvolvimento da solucao em formato de linha do tempo, destacando as principais decisoes e aprendizados ao longo do projeto.

## Linha do tempo do desenvolvimento

**1) Entendimento do problema e criterios**
- Interpretamos o desafio: identificar pedidos com dados pessoais que estavam classificados como publicos.
- Definimos o que conta como dado pessoal (ex.: nome, CPF, telefone, e-mail, endereco), alinhado a LGPD.

**2) Definicao da estrategia**
- Optamos por uma abordagem hibrida: regras deterministicas (regex) + modelo de deep learning.
- Objetivo: reduzir falsos negativos com regras fortes e melhorar cobertura com o modelo.

**3) Primeira camada: regras deterministicas (regex)**
- Criamos deteccao direta de PII estruturada (CPF, telefone, e-mail, CEP).
- Incluimos heuristicas de nome completo e enderecos com padrao do DF.

**4) Pre-processamento e generalizacao**
- Tratamos e normalizamos o texto para reduzir ruido e padronizar entradas.
- Buscamos evitar overfitting e melhorar a generalizacao para casos reais.

**5) Camada de modelo pre-treinado**
- Usamos um BERT em portugues para capturar padroes semanticos (ex.: nomes proprios e contexto).
- Ajustamos o uso de chunking para lidar com textos longos.

**6) Guardrails e desempate**
- Criamos um guardrail para casos individuais sem PII explicita (ex.: conta, protocolo).
- Adicionamos desempate quando o texto parece agregado e o score fica perto do limiar.

**7) Avaliacao e ajustes**
- Consideramos limitacoes de tempo e custo computacional.
- Ajustamos thresholds e heuristicas com foco em reduzir divulgacao indevida.

## Resultado
A solucao final combina interpretabilidade (regras claras) com poder de generalizacao (modelo), aumentando a confiabilidade na classificacao de pedidos e reduzindo o risco de exposicao de dados pessoais.

## Licoes aprendidas
- Combinar regras deterministicas com modelo melhora cobertura e reduz riscos.
- Pre-processamento consistente impacta diretamente na qualidade da inferencia.
- Guardrails simples evitam erros criticos em casos individuais.
- Escolher thresholds e desempates e um ajuste fino entre precisao e seguranca.
- Transparencia no fluxo ajuda na explicacao e na auditabilidade dos resultados.
alou
