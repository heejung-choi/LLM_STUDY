"""
rag_query.py

Qdrant에 업로드된 보험 약관 청크들을 벡터 검색하고,
검색 결과를 HTTP API 기반 Qwen LLM(OpenAI 호환 포맷)에 전달하여 답변을 스트리밍 생성

Usage:
    python rag_query.py
"""

import os
import sys
import json
import requests
import httpx
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# 1) .env 로드
load_dotenv()

# 2) 환경변수로부터 설정값 불러오기
QDRANT_HOST       = os.getenv("QDRANT_HOST")
QDRANT_PORT       = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL")
LLM_API_URL       = os.getenv("LLM_API_URL", "http://172.27.209.100:8000/v1/chat/completions")
LLM_MODEL_ID      = os.getenv("LLM_MODEL_ID", "/model/gen/Qwen2.5-7B-Instruct")
TOP_K             = int(os.getenv("TOP_K", 5))

# 필수 환경변수 확인
missing = [k for k in ("QDRANT_HOST", "QDRANT_COLLECTION", "EMBEDDING_API_URL","LLM_API_URL") if not os.getenv(k)]
if missing:
    print(f"ERROR: 다음 환경변수가 설정되어 있지 않습니다: {', '.join(missing)}", file=sys.stderr)
    sys.exit(1)

# 3) Qdrant 클라이언트 초기화
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
    return resp.json()["data"][0]["embedding"]

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

def stream_llm_response(prompt: str):
    """
    LLM 서버에 HTTP POST 요청을 보내고 stream 형식으로 응답 출력
    """
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
                    delta = json.loads(line.strip())
                    content = delta['choices'][0]['delta'].get('content', '')
                    print(content, end="", flush=True)

def answer_question(question: str):
    """
    질의를 임베딩 → Qdrant 검색 → 프롬프트 생성 → HTTP 기반 LLM 호출
    """
    q_vec = embed_query(question)
    docs = search_knn(q_vec)

    if not docs:
        print("🔍 관련된 정보를 찾을 수 없습니다.")
        return

    # 검색 결과 컨텍스트 구성
    context = []
    for hit in docs:
        payload = hit.payload
        ctx = f"{payload['chapter']} {payload['article']} (clause {payload['clause_index']}):\n{payload['text']}"
        context.append(ctx)
    joined = "\n\n---\n\n".join(context)

    # 최종 프롬프트 구성
    prompt = (
        "다음은 보험 약관에서 발췌한 내용입니다. 정확한 정보는 반드시 원문 약관을 확인하세요.\n\n"
        f"{joined}\n\n"
        f"### 질문: {question}\n\n"
        "### 답변:"
    )

    print("▶ 답변:")
    stream_llm_response(prompt)
    print("\n")  # 줄바꿈

if __name__ == "__main__":
    print("질문을 입력하세요 (종료는 Ctrl+C):")
    try:
        while True:
            q = input("> ").strip()
            if not q:
                continue
            answer_question(q)
    except KeyboardInterrupt:
        print("\n종료합니다.")
        sys.exit(0)
