import os
import sys
import json
import difflib
import requests
import httpx
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

# ── 환경변수 로드 및 필수값 검사 ─────────────────────────
load_dotenv()
QDRANT_HOST         = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT         = int(os.getenv("QDRANT_PORT", 6333))
PRODUCTS_COLLECTION = os.getenv("PRODUCTS_COLLECTION", "insurance_products")
POLICIES_COLLECTION = os.getenv("POLICIES_COLLECTION", "insurance_policies")
EMBEDDING_API_URL   = os.getenv("EMBEDDING_API_URL")
LLM_API_URL         = os.getenv("LLM_API_URL")
LLM_MODEL_ID        = os.getenv("LLM_MODEL_ID", "/model/gen/Midm")
TOP_K               = int(os.getenv("TOP_K", 5))

required = ("EMBEDDING_API_URL", "LLM_API_URL")
missing = [k for k in required if not os.getenv(k)]
if missing:
    print(f"ERROR: 환경변수 누락: {', '.join(missing)}", file=sys.stderr)
    sys.exit(1)

# ── Qdrant 클라이언트 초기화 ────────────────────────────
qdrant = QdrantClient(url=f"http://{QDRANT_HOST}:{QDRANT_PORT}", prefer_grpc=False)

def load_product_names() -> list[str]:
    """insurance_products 컬렉션에서 상품명(name) 목록 조회"""
    resp = qdrant.scroll(collection_name=PRODUCTS_COLLECTION, with_payload=True, limit=10000)
    points = resp[0] if isinstance(resp, tuple) else resp.points
    return [
        pt.payload["name"]
        for pt in points
        if pt.payload and pt.payload.get("name")
    ]

def load_policy_ids() -> list[str]:
    """insurance_policies 컬렉션에서 고유한 product_id 목록 조회 (is_common=True 제외)"""
    resp = qdrant.scroll(
        collection_name=POLICIES_COLLECTION,
        with_payload=True,
        limit=10000
    )
    points = resp[0] if isinstance(resp, tuple) else resp.points

    return sorted({
        pt.payload.get("product_id")
        for pt in points
        if pt.payload
           and pt.payload.get("product_id")
           and not pt.payload.get("is_common", False)   # 여기서 common 제외
    })

def find_best_match(user_input: str, choices: list[str], n: int = 3, cutoff: float = 0.35) -> list[str]:
    return difflib.get_close_matches(user_input, choices, n=n, cutoff=cutoff)

def embed_query(text: str) -> list[float]:
    resp = requests.post(EMBEDDING_API_URL, json={"input": [text]})
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]

def search_knn(query_vector: list[float], policy_id: str):
    """insurance_policies 컬렉션에서 product_id 필터로 KNN 검색 (is_common=True 제외)"""
    print(f"[DEBUG] Qdrant 검색 (policies) for product_id={policy_id!r}")
    flt = Filter(
        must=[
            FieldCondition(key="product_id", match=MatchValue(value=policy_id))
        ],
        must_not=[
            FieldCondition(key="is_common", match=MatchValue(value=True))
        ]
    )
    resp = qdrant.query_points(
        collection_name=POLICIES_COLLECTION,
        query=query_vector,
        limit=TOP_K,
        with_payload=True,
        query_filter=flt
    )
    return resp.points

def stream_llm_response(prompt: str):
    headers = {
        "Content-Type": "application/json",
        "Connection": "Keep-Alive",
    }
    payload = {
        "model": LLM_MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 1024,
        "stream": True,
    }
    with httpx.stream("POST", LLM_API_URL, json=payload, headers=headers, timeout=None) as ws:
        for line in ws.iter_lines():
            if not line:
                continue
            decoded = line.strip()
            if not decoded.startswith("data:"):
                continue
            data = decoded[len("data:"):].strip()
            if data == "[DONE]":
                print()
                return
            try:
                delta = json.loads(data)
            except json.JSONDecodeError:
                continue
            content = delta.get("choices")[0].get("delta", {}).get("content", "")
            print(content, end="", flush=True)

def answer_question(question: str, product_name: str):
    # 1) 상품명 → insurance_products에서 name 매칭
    products = load_product_names()
    if product_name not in products:
        print("선택한 상품이 목록에 없습니다.")
        return

    # 2) insurance_policies의 product_id 목록에서 유사한 ID 찾기
    policy_ids = load_policy_ids()
    matches = find_best_match(product_name, policy_ids, n=1, cutoff=0.4)
    if not matches:
        print("관련된 약관(컬렉션 내 product_id)이 없습니다.")
        return
    selected_policy_id = matches[0]
    print(f"[INFO] 매핑된 policy_id: {selected_policy_id}")

    # 3) 질문 임베딩 → 선택된 policy_id로 KNN 검색
    vec = embed_query(question)
    docs = search_knn(vec, selected_policy_id)
    if not docs:
        print("관련된 약관 청크를 찾을 수 없습니다.")
        return

    # 4) 검색 결과 컨텍스트 구성 & LLM 호출
    context = []
    for hit in docs:
        p = hit.payload or {}
        ctx = (
            f"{p.get('chapter')} {p.get('article')} "
            f"(clause {p.get('clause_index')}):\n{p.get('text')}"
        )
        context.append(ctx)
    joined = "\n\n---\n\n".join(context)
    prompt = (
        "다음은 보험 약관에서 발췌한 내용입니다. 정확한 정보는 반드시 원문 약관을 확인하세요.\n\n"
        f"{joined}\n\n"
        f"### 질문: {question}\n"
        "### 답변 (보험 약관 전문가로서 구체적이고 상세하게, 예시를 들어 설명해주세요):"
    )
    print("[DEBUG] LLM PROMPT:")
    #print(prompt)
    print("────────────────────────────────────────────────────────")
    stream_llm_response(prompt)

def list_uploaded_products():
    print("[INFO] 업로드된 상품 목록 조회 중...")
    names = load_product_names()
    print(f"[INFO] 총 {len(names)}개 상품:")
    for name in names:
        print(" -", name)

if __name__ == "__main__":
    list_uploaded_products()
    print("상품명을 포함한 질문을 입력하세요 (예: '희망파트너 변액유니버셜의 보장내용이 뭐야?')")
    try:
        products = load_product_names()
        while True:
            # 1) 상품명 포함 질문 → 상품 선택
            q = input("> ").strip()
            if not q:
                continue
            matches = find_best_match(q, products)
            # 유사한 상품명이 없을 때, 상품명을 모두 번호와 함께 보여줍니다.
            if not matches:
                print("등록된 상품명 리스트입니다. 이 중 하나로 선택해주세요:")
                for idx, name in enumerate(products, start=1):
                    print(f" {idx}. {name}")
                # 새 질문 받기 위해 루프 계속
                continue
            print("질문하신 것과 가장 유사한 상품명 리스트 입니다.")
            for i, name in enumerate(matches, 1):
                print(f" {i}. {name}")
            sel = input("답변을 원하시는 상품명 번호 선택을 하시거나 Enter를 눌러 취소해주세요.").strip()
            if not sel.isdigit() or not (1 <= int(sel) <= len(matches)):
                continue
            chosen = matches[int(sel) - 1]
            print(f"선택된 상품: {chosen}\n")

            # 2) 동일 상품에 대한 추가 질문 루프 (예/아니오만 허용)
            while True:
                answer_question(q, chosen)
                # “예” 또는 “아니오”만 입력받도록 반복
                while True:
                    more = input("동일 상품에 대한 추가 질문이 있으신가요? (예/아니오): ").strip()
                    if more in ("예", "아니오"):
                        break
                    print("‘예’ 또는 ‘아니오’ 중 하나만 입력해주세요.")
                if more == "예":
                    q = input(f"{chosen}에 대한 추가 질문을 입력하세요: ").strip()
                    if not q:
                        continue
                    # 동일 상품으로 다시 질문
                    continue
                # “아니오”면 상품 선택 단계로 반환
                break

            # 3) ‘아니오’ 시 새 질문 안내 및 종료 처리
            print("상품명을 포함한 질문을 입력하세요 (예: '희망파트너 변액유니버셜의 보장내용이 뭐야?'). " \
            "더이상 질문이 없을 경우 '종료' 라고 입력해주세요.")
            new_q = input("> ").strip()
            if new_q == "종료":
                print("종료합니다.")
                sys.exit(0)
            # 종료가 아닌 경우 다시 상품 선택 로직으로 돌아갑니다.
    except KeyboardInterrupt:
        print("종료합니다.")
        sys.exit(0)
