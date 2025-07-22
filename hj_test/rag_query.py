"""
rag_query.py

Qdrant에 업로드된 보험 약관 청크들을 벡터 검색하고,
검색 결과를 Hugging Face LLM(OpenAI API 호환)을 통해 답변을 생성하는 예제 스크립트입니다.

Usage:
    python rag_query.py
"""

import os
import sys
import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from openai import OpenAI

# 1) .env 로드
load_dotenv()

# 2) 환경변수로부터 설정값 불러오기
QDRANT_HOST       = os.getenv("QDRANT_HOST")
QDRANT_PORT       = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL")
HF_API_KEY        = os.getenv("HF_TOKEN")
HF_MODEL_ID       = os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
TOP_K             = int(os.getenv("TOP_K", 5))

# 필수 환경변수 확인
missing = [k for k in ("QDRANT_HOST","QDRANT_COLLECTION","EMBEDDING_API_URL","HF_TOKEN") if not os.getenv(k)]
if missing:
    print(f"ERROR: 다음 환경변수가 설정되어 있지 않습니다: {', '.join(missing)}", file=sys.stderr)
    sys.exit(1)

# 3) Hugging Face (OpenAI-compatible) 클라이언트 설정
client = OpenAI(
    base_url="https://api.endpoints.huggingface.cloud/v1",
    api_key=HF_API_KEY,
)

# 4) Qdrant 클라이언트 초기화
qdrant = QdrantClient(
    url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
    prefer_grpc=False
)

def embed_query(query: str) -> list[float]:
    """
    사용자의 질의를 embedding API에 보내고 벡터를 리턴합니다.
    """
    payload = {"input": [query]}
    resp = requests.post(EMBEDDING_API_URL, json=payload)
    resp.raise_for_status()
    data = resp.json()

    return data["data"][0]["embedding"]

def search_knn(query_vector: list[float], top_k: int = TOP_K):
    """
    Qdrant에서 k-NN 검색을 수행하여 유사 문서들을 리턴합니다.
    """
    hits = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )
    return hits.points  

def answer_question(question: str) -> str:
    """
    질의를 임베딩 → Qdrant 검색 → Hugging Face ChatCompletion으로 답변 생성
    """
    # 1) 질의 임베딩
    q_vec = embed_query(question)

    # 2) k-NN 검색
    docs = search_knn(q_vec, top_k=TOP_K)

    if not docs:
        return "관련된 정보를 찾을 수 없습니다. 질문을 다시 확인해 주세요."

    # 3) 검색 결과를 Prompt용 컨텍스트로 조합
    context = []
    for hit in docs:
        payload = hit.payload
        # payload 에 id, chapter, article, text, clause_index 등이 담겨있다고 가정
        ctx = f"{payload['chapter']} {payload['article']} (clause {payload['clause_index']}):\n{payload['text']}"
        context.append(ctx)
    joined = "\n\n---\n\n".join(context)

    # 4) Hugging Face 호출 (OpenAI 호환 포맷)
    prompt = (
        "다음은 보험 약관에서 발췌한 내용입니다. 정확한 정보는 반드시 원문 약관을 확인하세요.\n\n"
        f"{joined}\n\n"
        f"### 질문: {question}\n\n"
        "### 답변:"
    )
    resp = client.chat.completions.create(
        model=HF_MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()

if __name__ == "__main__":
    print("질문을 입력하세요 (종료는 Ctrl+C):")
    try:
        while True:
            q = input("> ").strip()
            if not q:
                continue
            print("\n▶ 답변:\n", answer_question(q), "\n")
    except KeyboardInterrupt:
        print("\n종료합니다.")
        sys.exit(0)
