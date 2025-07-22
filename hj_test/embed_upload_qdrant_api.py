import os
import sys
import json
import time
import uuid
import requests
from tqdm import tqdm
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

# ── 환경변수 로드 ─────────────────────────────────────────
load_dotenv()
QDRANT_HOST       = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT       = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "insurance_policies")
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL")
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large-instruct")

if not EMBEDDING_API_URL:
    print("ERROR: .env에 EMBEDDING_API_URL이 설정되어 있지 않습니다.", file=sys.stderr)
    sys.exit(1)

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# 임베딩 API 호출: 텍스트 리스트 → 벡터 리스트
def get_embeddings(texts: list[str]) -> list[list[float]]:
    payload = {"model": EMBEDDING_MODEL, "input": texts}
    r = requests.post(EMBEDDING_API_URL, json=payload)
    r.raise_for_status()
    data = r.json()
    items = data.get("data") or data.get("result")
    return [itm["embedding"] if isinstance(itm, dict) and "embedding" in itm else itm for itm in items]

# 약관 JSONL 디렉토리 처리: 각 청크를 벡터화 후 Qdrant에 업로드
def process_jsonl_folder(folder: str):
    print(f"\n약관 JSONL 디렉토리 처리 시작: {folder}")
    files = []
    for root, _, fnames in os.walk(folder):
        for fn in fnames:
            if fn.lower().endswith(".jsonl"):
                files.append(os.path.join(root, fn))
    if not files:
        print("ERROR: JSONL 파일을 찾을 수 없습니다.", file=sys.stderr)
        sys.exit(1)

    print(f"총 {len(files)}개 JSONL 파일 처리\n")

    # ⚠️ 기존 insurance_policies 컬렉션 삭제 후 재생성
    client.recreate_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=rest.VectorParams(size=1024, distance=rest.Distance.COSINE)
    )
    print(f"'{QDRANT_COLLECTION}' 컬렉션 재생성 완료")

    for path in files:
        rel = os.path.relpath(path, folder)
        print(f"▶ {rel} → ", end="")
        docs = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        print(f"{len(docs)}개 청크 임베딩/업로드")

        batch = 16
        for i in tqdm(range(0, len(docs), batch), desc="임베딩 중"):
            chunk = docs[i : i+batch]
            texts = [d["text"] for d in chunk]
            try:
                embs = get_embeddings(texts)
            except Exception as e:
                print(f"\nERROR: 임베딩 실패: {e}", file=sys.stderr)
                continue

            points = []
            for d, vec in zip(chunk, embs):
                payload = {
                    "id":           d.get("id"),
                    "chapter":      d.get("chapter"),
                    "article":      d.get("article"),
                    "clause_index": d.get("clause_index"),
                    "text":         d.get("text"),
                    "source_file":  os.path.basename(path),
                }
                points.append(
                    rest.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vec,
                        payload=payload
                    )
                )
            client.upsert(collection_name=QDRANT_COLLECTION, points=points)
            time.sleep(0.05)
        print("완료\n")

#  보험 상품명 JSON → Qdrant 벡터 업로드 (insurance_products 컬렉션)
def embed_upload_insurance_products(json_path: str, collection_name: str = "insurance_products"):
    if not os.path.isfile(json_path):
        print(f"ERROR: 파일이 존재하지 않습니다: {json_path}", file=sys.stderr)
        return

    print(f"\n보험 상품명 JSON 로딩 중: {json_path}")
    with open(json_path, encoding="utf-8") as f:
        items = json.load(f)

    names = [item["PRDCD_NAM"] for item in items]
    try:
        vectors = get_embeddings(names)
    except Exception as e:
        print(f"ERROR: 임베딩 실패: {e}", file=sys.stderr)
        return

    print(f"총 {len(names)}개 상품명을 '{collection_name}' 컬렉션에 업로드합니다.")
    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        # ⚠️ 기존 insurance_products 컬렉션 삭제
        client.delete_collection(collection_name=collection_name)
        print(f"기존 '{collection_name}' 컬렉션 삭제 완료")

    client.create_collection(
        collection_name=collection_name,
        vectors_config=rest.VectorParams(
            size=len(vectors[0]),
            distance=rest.Distance.COSINE
        )
    )
    

    points = []
    for item, vec in zip(items, vectors):
        payload = {
            "name": item["PRDCD_NAM"],
            "pk": item["PK"],
            "class": item.get("CLASSNM"),
            "start": item.get("SALE_START_DT"),
            "end": item.get("SALE_END_DT"),
        }
        points.append(
            rest.PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload=payload
            )
        )

    client.upsert(collection_name=collection_name, points=points)
    print(f"'{collection_name}' 업로드 완료 ({len(points)}건)")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python embed_upload_qdrant_api.py <jsonl_dir|insurance_products.json>", file=sys.stderr)
        sys.exit(1)

    arg = sys.argv[1]
    if arg.lower().endswith(".json"):
        embed_upload_insurance_products(arg)
    elif os.path.isdir(arg):
        process_jsonl_folder(arg)
        print("모두 완료되었습니다!")
    else:
        print(f"ERROR: 유효한 디렉터리나 JSON 파일이 아닙니다: {arg}", file=sys.stderr)
        sys.exit(1)