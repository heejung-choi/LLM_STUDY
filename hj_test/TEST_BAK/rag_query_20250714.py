import os
import sys
import json
import difflib
import requests
import httpx
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

# 1) .env 로드
load_dotenv()

# 2) 환경변수 불러오기
QDRANT_HOST       = os.getenv("QDRANT_HOST")
QDRANT_PORT       = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL")
LLM_API_URL       = os.getenv("LLM_API_URL")
LLM_MODEL_ID      = os.getenv("LLM_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
TOP_K             = int(os.getenv("TOP_K", 5))
PRODUCTS_FILE     = "insurance_products.json"

# 필수 환경변수 확인
missing = [k for k in ("QDRANT_HOST", "QDRANT_COLLECTION", "EMBEDDING_API_URL", "LLM_API_URL") if not os.getenv(k)]
if missing:
    print(f"ERROR: 다음 환경변수가 설정되어 있지 않습니다: {', '.join(missing)}", file=sys.stderr)
    sys.exit(1)

# 3) Qdrant 클라이언트 초기화
qdrant = QdrantClient(
    url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
    prefer_grpc=False
)

def load_product_names():
    with open(PRODUCTS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return [item["PRDCD_NAM"] for item in data]

def find_best_match(user_input, product_names):
    return difflib.get_close_matches(user_input, product_names, n=3, cutoff=0.5)

def embed_query(query):
    payload = {"input": [query]}
    resp = requests.post(EMBEDDING_API_URL, json=payload)
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]

def search_knn(query_vector, product_name):
    hits = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        vector=query_vector,
        limit=TOP_K,
        with_payload=True,
        query_filter=rest.Filter(
            must=[
                rest.FieldCondition(
                    key="source_file",
                    match=rest.MatchValue(value=f"{product_name}.jsonl")
                )
            ]
        )
    )
    return hits.points

def stream_llm_response(prompt: str):
    headers = {
        "Content-Type": "application/json",
        "Connection": "Keep-Alive"
    }
    payload = {
        "model": LLM_MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 1024,
        "stream": True
    }
    with httpx.stream("POST", LLM_API_URL, json=payload, headers=headers, timeout=None) as response:
        for chunk in response.iter_text():
            if chunk.strip():
                for line in chunk.split("data: "):
                    if not line.strip() or line.strip() == "[DONE]":
                        continue
                    try:
                        delta = json.loads(line.strip())
                        content = delta['choices'][0]['delta'].get('content', '')
                        print(content, end="", flush=True)
                    except Exception:
                        continue

def answer_question(question, product_name):
    q_vec = embed_query(question)
    docs = search_knn(q_vec, product_name)
    if not docs:
        return "🔍 해당 상품의 약관에서 유사한 내용을 찾을 수 없습니다."

    context = []
    for hit in docs:
        p = hit.payload
        ctx = f"{p['chapter']} {p['article']} (clause {p['clause_index']}):\n{p['text']}"
        context.append(ctx)
    joined = "\n\n---\n\n".join(context)

    prompt = (
        "请用韩语回答以下问题。请注意，不要使用中文、英文或其他语言，只能使用韩语。如果你使用了其他语言，将被视为错误。\n\n"
        "다음은 보험 약관에서 발췌한 내용입니다. 정확한 정보는 반드시 원문 약관을 확인하세요.\n\n"
        f"{joined}\n\n"
        f"### 질문: {question}\n\n"
        "### 답변:"
    )

    stream_llm_response(prompt)

if __name__ == "__main__":
    product_names = load_product_names()
    print("상품명을 포함한 질문을 입력하세요 (예: 'iM 하이브리드연금보험의 보장 내용이 뭐야?')")
    try:
        while True:
            q = input("> ").strip()
            if not q:
                continue

            best_matches = find_best_match(q, product_names)
            if not best_matches:
                print("❌ 입력된 질문에서 유사한 보험 상품명을 찾을 수 없습니다. 다시 입력해주세요.")
                continue

            print("🔍 가장 유사한 상품명:", best_matches[0])
            yn = input(f"👉 이 상품에 대해 질문한 것이 맞나요? (Y/N): ").strip().lower()
            if yn == "y":
                print("\n▶ 답변:")
                answer_question(q, best_matches[0])
                print("\n")
            else:
                print("📌 유사한 상품명 후보:")
                for p in best_matches:
                    print("-", p)
                print("다시 정확한 상품명을 포함하여 질문해주세요.\n")
    except KeyboardInterrupt:
        print("\n종료합니다.")
        sys.exit(0)
