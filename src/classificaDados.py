# predict3.py
import argparse
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertTokenizer

# ==========================================
# CONFIGURACOES FIXAS (sem arg p/ threshold)
# ==========================================
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "bert_finetuned")

THRESHOLD_PRIVADO = 0.30
STRONG_PRIVADO = 0.55


# ==========================================
# PREPROCESS (alinhado ao treino)
# ==========================================
# Funcao: prepara o texto para o modelo (remove espacos extras e troca quebra de linha por [SEP]).
# Entrada: text (str) com o texto bruto digitado ou lido do arquivo.
# Saida: texto limpo (str) pronto para tokenizacao.
def clean_text_for_inference(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\n", " [SEP] ").replace("\t", " ")
    text = " ".join(text.split())
    return text


# ==========================================
# HEURISTICAS
# ==========================================
# Funcao: verifica se o texto tem linguagem de 1a pessoa (ex.: "minha", "meu").
# Entrada: tl (str) em minusculo para facilitar a busca por termos.
# Saida: True se achar 1a pessoa, False caso contrario.
def _first_person_anchor(tl: str) -> bool:
    return any(x in tl for x in [
        "meu ", "minha ", "meus ", "minhas ",
        "me mudei", "sou inquilina", "sou inquilino",
        "minha residência", "minha residencia",
        "meu imóvel", "meu imovel",
        "na minha casa", "na minha residência", "na minha residencia",
        "no meu endereço", "no meu endereco",
    ])


# Funcao: procura sinais de pedido geral/agregado (ex.: "quantos", "percentual").
# Entrada: tl (str) em minusculo.
# Saida: True se parecer consulta agregada, False caso contrario.
def _aggregate_public_hint(tl: str) -> bool:
    """
    Indícios fortes de consulta institucional/agregada.
    Usado apenas como desempate para reduzir FP quando o modelo está "em cima do muro".
    """
    aggregate_terms = [
        "quantos", "quantas", "quantidade", "percentual", "média", "media",
        "estatística", "estatisticas", "estatísticas",
        "relatório", "relatorio", "consolidado", "dados agregados",
        "por mês", "por mes", "mensal", "trimestre", "semestre", "ano",
        "série histórica", "serie historica",
        "lista de", "tabela", "ranking",
        "procedimento geral", "regras e prazos", "norma", "como funciona",
        "apenas informações gerais", "apenas informacoes gerais",
        "sem identificação", "sem identificacao", "sem identificar",
        "dados anonimizados", "anonimizado",
    ]
    return any(term in tl for term in aggregate_terms)


# Funcao: aplica regras fixas de PII para decidir privado sem usar o modelo.
# Entrada: text (str) original com possiveis dados pessoais.
# Saida: (bool, motivo) onde motivo explica a regra que disparou.
def deterministic_precheck_privado(text: str) -> Tuple[bool, Optional[str]]:
    """
    Pré-check determinístico de ALTA confiança para reduzir FN.
    Não depende do BERT.
    """
    if not isinstance(text, str):
        return False, None

    t = text.strip()
    tl = t.lower()
    if not t:
        return False, None

    cpf_regex = r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b|\b\d{11}\b"
    email_regex = r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b"
    telefone_regex = r"\b(?:\+?\d{1,3}\s*)?(?:\(?\d{2}\)?\s*)?\d{4,5}-?\d{4}\b"
    cep_regex = r"\b\d{5}-?\d{3}\b"

    if re.search(cpf_regex, t, flags=re.IGNORECASE):
        return True, "det:cpf"
    if re.search(email_regex, t, flags=re.IGNORECASE):
        return True, "det:email"
    if re.search(telefone_regex, t, flags=re.IGNORECASE):
        return True, "det:telefone"
    if re.search(cep_regex, t, flags=re.IGNORECASE):
        return True, "det:cep"

    nome_completo = (
        r"([A-ZÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛÇ][a-záàâãéèêíìîóòôõúùûç'\-]+"
        r"(?:\s+[A-ZÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛÇ][a-záàâãéèêíìîóòôõúùûç'\-]+){1,4})"
    )

    assinatura_regex = rf"\b(atenciosamente|cordialmente|att\.?|at\.?|ass\.?|ass:|assinatura:)\b[\s,:-]*{nome_completo}\b"
    meu_nome_regex = rf"\b(meu nome é|me chamo)\b[\s,:-]*{nome_completo}\b"

    blacklist_last = {"gostaria", "solicito", "venho", "preciso", "quero", "informo", "requeiro"}

    m = re.search(assinatura_regex, t, flags=re.IGNORECASE)
    if m:
        nome = m.group(1)
        last = nome.split()[-1].lower()
        if last not in blacklist_last:
            return True, "det:assinatura_nome"

    m2 = re.search(meu_nome_regex, t, flags=re.IGNORECASE)
    if m2:
        nome = m2.group(1)
        last = nome.split()[-1].lower()
        if last not in blacklist_last:
            return True, "det:meu_nome"

    df_siglas = r"\b(sqn|sqs|shis|shcg|cln|cls|crn|crs|qn|qi|qnl|qne|qng|qnp|qnr|qnm|qnd|qna|qsa|qsb|qsc|qsd)\b"
    end_elem = r"\b(bloco|lote|conjunto|apto|apartamento|casa|loja|edif[ií]cio|quadra|setor|rua|avenida|av\.?)\b"
    num = r"\b\d{1,4}\b"

    has_df_sigla = re.search(df_siglas, tl, flags=re.IGNORECASE) is not None
    has_end_elem = re.search(end_elem, tl, flags=re.IGNORECASE) is not None
    has_num = re.search(num, t) is not None

    personal_anchor = any(x in tl for x in [
        "sou inquilina", "sou inquilino", "resido", "moro",
        "minha residência", "minha residencia",
        "meu imóvel", "meu imovel",
        "imóvel localizado", "imovel localizado",
        "localizado na", "localizada na",
    ])

    if has_df_sigla and has_end_elem and has_num:
        return True, "det:endereco_df"
    if personal_anchor and (has_df_sigla or has_end_elem) and has_num:
        return True, "det:endereco_pessoal"

    return False, None


# Funcao: detecta pedido individual mesmo sem PII explicita (ex.: conta, protocolo).
# Entrada: text (str) original.
# Saida: (bool, motivo) indicando se deve marcar como privado.
def guardrail_caso_individual(text: str) -> Tuple[bool, Optional[str]]:
    """
    Guardrail: caso individual (identificação indireta), sem PII explícito.
    Só dispara com âncora de 1ª pessoa + termos de caso individual.
    """
    if not isinstance(text, str):
        return False, None
    t = text.strip()
    tl = t.lower()
    if not t:
        return False, None

    first_person = _first_person_anchor(tl)

    caso_terms = [
        "conta de água", "conta de agua", "titularidade", "hidrômetro", "hidrometro",
        "fatura", "boleto", "consumo", "leitura do consumo", "cobrança", "cobranca",
        "matrícula", "matricula", "cadastro", "atualização cadastral", "atualizacao cadastral",
        "benefício", "beneficio", "auxílio", "auxilio", "dependente", "repasse", "mensalidade",
        "prontuário", "prontuario", "laudo", "exame", "hospital", "tratamento",
        "aposentadoria", "pensão", "pensao",
        "protocolo", "andamento", "processo", "sei",
        "segunda via", "transferência de titularidade", "transferencia de titularidade",
        "contrato em questão", "contrato em questao", "anexo", "anexei", "anexado",
    ]

    if first_person and any(term in tl for term in caso_terms):
        if _aggregate_public_hint(tl):
            return False, None
        return True, "guardrail:caso_individual_1p"

    return False, None


# ==========================================
# CHUNKING POR IDs
# ==========================================
# Funcao: corta os ids em janelas sobrepostas para textos longos.
# Entrada: lista de ids, tamanho maximo da janela, overlap e limite opcional de chunks.
# Saida: lista de listas de ids (cada uma e um chunk).
def chunk_ids(
    input_ids: List[int],
    max_length: int,
    overlap: int,
    max_chunks: Optional[int] = None,
) -> List[List[int]]:
    if max_length <= 0:
        raise ValueError("max_length must be > 0")
    if overlap < 0 or overlap >= max_length:
        raise ValueError("overlap must be in [0, max_length-1]")

    step = max_length - overlap
    if step <= 0:
        raise ValueError("Invalid step computed; check max_length/overlap.")

    chunks: List[List[int]] = []
    start = 0
    n = len(input_ids)

    if n == 0:
        return [[]]

    while start < n:
        end = min(start + max_length, n)
        chunks.append(input_ids[start:end])

        if max_chunks is not None and len(chunks) >= max_chunks:
            break
        if end == n:
            break

        start += step

    return chunks


# Funcao: transforma um chunk em um trecho curto legivel para auditoria.
# Entrada: tokenizer, ids do chunk e max_chars (int).
# Saida: string curta com parte do texto decodificado.
def decode_chunk_snippet(tokenizer: BertTokenizer, ids: List[int], max_chars: int = 200) -> str:
    if not ids:
        return ""
    txt = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    txt = txt.replace("\n", " ")
    return txt[:max_chars]


# ==========================================
# PREDICAO POR DOCUMENTO
# ==========================================
# Funcao: roda o fluxo completo (pre-check -> guardrail -> modelo -> desempate).
# Entrada: texto, tokenizer, modelo, device e parametros de batch/chunk.
# Saida: dict com scores, rotulo final e detalhes do melhor chunk.
def predict_document(
    text: str,
    tokenizer: BertTokenizer,
    model: BertForSequenceClassification,
    device: torch.device,
    batch_size: int,
    max_length_chunk: int,
    overlap: int,
    max_chunks: Optional[int],
) -> Dict[str, Any]:
    text = clean_text_for_inference(text)

    if not text:
        return {
            "doc_pred_privado": False,
            "score_privado": 0.0,
            "score_publico": 1.0,
            "best_chunk_index": -1,
            "best_chunk_score_privado": 0.0,
            "best_chunk_snippet": "",
            "num_chunks": 0,
            "decision_reason": "empty_input",
        }

    is_priv, reason = deterministic_precheck_privado(text)
    if is_priv:
        return {
            "doc_pred_privado": True,
            "score_privado": 0.999,
            "score_publico": 0.001,
            "best_chunk_index": -1,
            "best_chunk_score_privado": 0.999,
            "best_chunk_snippet": f"[PRECHECK] {reason}",
            "num_chunks": 0,
            "decision_reason": reason,
        }

    gr_priv, gr_reason = guardrail_caso_individual(text)
    if gr_priv:
        return {
            "doc_pred_privado": True,
            "score_privado": 0.900,
            "score_publico": 0.100,
            "best_chunk_index": -1,
            "best_chunk_score_privado": 0.900,
            "best_chunk_snippet": f"[GUARDRAIL] {gr_reason}",
            "num_chunks": 0,
            "decision_reason": gr_reason,
        }

    encoded_full = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_tensors=None,
    )
    full_ids: List[int] = encoded_full.get("input_ids", []) or []
    if not full_ids:
        return {
            "doc_pred_privado": False,
            "score_privado": 0.0,
            "score_publico": 1.0,
            "best_chunk_index": -1,
            "best_chunk_score_privado": 0.0,
            "best_chunk_snippet": "",
            "num_chunks": 0,
            "decision_reason": "empty_input",
        }

    chunks_ids = chunk_ids(
        input_ids=full_ids,
        max_length=max_length_chunk,
        overlap=overlap,
        max_chunks=max_chunks,
    )

    features = [{"input_ids": ids} for ids in chunks_ids]

    scores_privado_chunks: List[float] = []
    best_chunk_score_privado = -1.0
    best_chunk_index = -1
    best_chunk_ids: List[int] = []

    for i in range(0, len(features), batch_size):
        batch_features = features[i: i + batch_size]

        batch = tokenizer.pad(batch_features, padding=True, return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)
            probs = F.softmax(outputs.logits, dim=1)

        batch_scores_privado = probs[:, 1].detach().cpu().tolist()
        scores_privado_chunks.extend(batch_scores_privado)

        for j, p in enumerate(batch_scores_privado):
            idx = i + j
            if p > best_chunk_score_privado:
                best_chunk_score_privado = float(p)
                best_chunk_index = idx
                best_chunk_ids = chunks_ids[idx] if idx < len(chunks_ids) else []

    score_privado = max(scores_privado_chunks) if scores_privado_chunks else 0.0
    score_publico = 1.0 - score_privado

    doc_pred_privado = score_privado >= THRESHOLD_PRIVADO
    decision_reason = "model"

    tl = text.lower()
    if doc_pred_privado and score_privado < STRONG_PRIVADO:
        if _aggregate_public_hint(tl) and not _first_person_anchor(tl):
            doc_pred_privado = False
            decision_reason = "tie_break:aggregate_public"

    best_chunk_snippet = decode_chunk_snippet(tokenizer, best_chunk_ids, max_chars=200)

    return {
        "doc_pred_privado": doc_pred_privado,
        "score_privado": score_privado,
        "score_publico": score_publico,
        "best_chunk_index": best_chunk_index,
        "best_chunk_score_privado": best_chunk_score_privado,
        "best_chunk_snippet": best_chunk_snippet,
        "num_chunks": len(chunks_ids),
        "decision_reason": decision_reason,
    }


# ==========================================
# CLI
# ==========================================
# Funcao: modo interativo no terminal para testar frases uma a uma.
# Entrada: parametros de inferencia (batch, tamanho de chunk, overlap, max_chunks).
# Saida: nao retorna valor; apenas imprime resultados na tela.
def predict_manual(batch_size: int, max_length_chunk: int, overlap: int, max_chunks: Optional[int]) -> None:
    print("Loading trained model...")

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded on device: {device}")
    print(f"Threshold privado fixo: {THRESHOLD_PRIVADO:.2f}")

    print("\n--- MANUAL INFERENCE (type 'sair' to quit) ---")

    while True:
        text = input("\nDigite a frase para classificar: ")
        if text.lower() == "sair":
            break

        result = predict_document(
            text=text,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=batch_size,
            max_length_chunk=max_length_chunk,
            overlap=overlap,
            max_chunks=max_chunks,
        )

        pred_label = 1 if result["doc_pred_privado"] else 0
        pred_text = "privado" if pred_label == 1 else "publico"

        print(f"Resultado: {pred_text}")
        print(
            f"score_privado: {result['score_privado']:.4f} | "
            f"score_publico: {result['score_publico']:.4f} | "
            f"chunks={result['num_chunks']} | reason={result['decision_reason']}"
        )
        print(
            "Melhor chunk: "
            f"idx={result['best_chunk_index']} "
            f"score_privado={result['best_chunk_score_privado']:.4f} "
            f"trecho='{result['best_chunk_snippet']}'"
        )


# Funcao: processa uma planilha Excel e grava as predicoes em outro arquivo.
# Entrada: caminhos de entrada/saida, nome da coluna e parametros de inferencia.
# Saida: nao retorna valor; salva o arquivo de saida no disco.
def predict_xlsx(
    input_path: str,
    output_path: str,
    text_column: Optional[str] = "Texto Mascarado",
    batch_size: int = 32,
    max_length_chunk: int = 384,
    overlap: int = 128,
    max_chunks: Optional[int] = None,
) -> None:
    print("Loading model...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded on: {device}")
    print(f"Threshold privado fixo: {THRESHOLD_PRIVADO:.2f}")

    df = pd.read_excel(input_path)
    if df.empty:
        raise ValueError("Input Excel is empty.")

    if text_column is None:
        text_column = df.columns[0]
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in the Excel file.")

    texts = df[text_column].fillna("").astype(str).tolist()

    labels: List[int] = []
    pred_texts: List[str] = []
    score_privado_list: List[float] = []
    score_publico_list: List[float] = []
    best_chunk_index_list: List[int] = []
    best_chunk_score_privado_list: List[float] = []
    best_chunk_snippet_list: List[str] = []
    num_chunks_list: List[int] = []
    decision_reason_list: List[str] = []

    for text in texts:
        result = predict_document(
            text=text,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=batch_size,
            max_length_chunk=max_length_chunk,
            overlap=overlap,
            max_chunks=max_chunks,
        )

        label = 1 if result["doc_pred_privado"] else 0
        labels.append(label)
        pred_texts.append("privado" if label == 1 else "publico")

        score_privado_list.append(float(result["score_privado"]))
        score_publico_list.append(float(result["score_publico"]))
        best_chunk_index_list.append(int(result["best_chunk_index"]))
        best_chunk_score_privado_list.append(float(result["best_chunk_score_privado"]))
        best_chunk_snippet_list.append(result["best_chunk_snippet"])
        num_chunks_list.append(int(result["num_chunks"]))
        decision_reason_list.append(str(result["decision_reason"]))

    df["pred_label"] = labels
    df["pred_text"] = pred_texts
    df["score_privado"] = score_privado_list
    df["score_publico"] = score_publico_list
    df["best_chunk_index"] = best_chunk_index_list
    df["best_chunk_score_privado"] = best_chunk_score_privado_list
    df["best_chunk_snippet"] = best_chunk_snippet_list
    df["num_chunks"] = num_chunks_list
    df["decision_reason"] = decision_reason_list

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_excel(output_path, index=False)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict v3 (precheck + guardrail + tie-break).")
    parser.add_argument("--xlsx", help="Path to input .xlsx for batch prediction.")
    parser.add_argument("--output", help="Path to output .xlsx with predictions.")
    parser.add_argument("--text-column", help="Name of the text column in the Excel file.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for prediction.")
    parser.add_argument("--max-length-chunk", type=int, default=384, help="Max tokens per chunk.")
    parser.add_argument("--overlap", type=int, default=128, help="Token overlap between chunks.")
    parser.add_argument("--max-chunks", type=int, default=None, help="Optional cap on number of chunks.")
    args = parser.parse_args()

    if args.xlsx:
        if not args.output:
            raise SystemExit("--output is required when using --xlsx")
        predict_xlsx(
            args.xlsx,
            args.output,
            text_column=args.text_column,
            batch_size=args.batch_size,
            max_length_chunk=args.max_length_chunk,
            overlap=args.overlap,
            max_chunks=args.max_chunks,
        )
    else:
        predict_manual(
            batch_size=args.batch_size,
            max_length_chunk=args.max_length_chunk,
            overlap=args.overlap,
            max_chunks=args.max_chunks,
        )
