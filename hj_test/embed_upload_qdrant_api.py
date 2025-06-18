"""
embed_upload_qdrant_api.py
JSONL 청크 데이터를 임베딩 API → Qdrant 업로드

Usage:
    python embed_upload_qdrant_api.py <jsonl_dir>
"""

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
from qdrant_client import QdrantClient
#from langchain_huggingface.embeddings import HuggingFaceEmbeddings

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

# ── Qdrant 클라이언트 초기화 ───────────────────────────────
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# 컬렉션 삭제(초기화)
client.delete_collection(collection_name="insurance_policies")
print("insurance_policies 컬렉션을 삭제했습니다.")

def ensure_collection():
    names = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in names:
        print(f"NOTICE: Collection '{QDRANT_COLLECTION}' 생성 중…")
        client.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=rest.VectorParams(size=1024, distance=rest.Distance.COSINE)
        )
    else:
        print(f"OK: Collection '{QDRANT_COLLECTION}' 확인됨")

def get_embeddings(texts: list[str]) -> list[list[float]]:
    payload = {
        "model": EMBEDDING_MODEL,
        "input": texts
    }
    r = requests.post(EMBEDDING_API_URL, json=payload)
    r.raise_for_status()
    data = r.json()
    # OpenAI 스타일: data["data"] = [ { "embedding": [...] }, ... ]
    items = data.get("data") or data.get("result")
    return [
        itm["embedding"] if isinstance(itm, dict) and "embedding" in itm else itm
        for itm in items
    ]

def process_jsonl_folder(folder: str):
    # 1) .jsonl 파일 수집
    files = []
    for root, _, fnames in os.walk(folder):
        for fn in fnames:
            if fn.lower().endswith(".jsonl"):
                files.append(os.path.join(root, fn))
    if not files:
        print("ERROR: JSONL 파일을 찾을 수 없습니다.", file=sys.stderr)
        sys.exit(1)

    print(f"총 {len(files)}개 JSONL 파일 처리\n")

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

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python embed_upload_qdrant_api.py <jsonl_dir>", file=sys.stderr)
        sys.exit(1)
    folder = sys.argv[1]
    if not os.path.isdir(folder):
        print(f"ERROR: '{folder}' 디렉터리가 없습니다.", file=sys.stderr)
        sys.exit(1)

    print("Qdrant 연결 중…")
    ensure_collection()
    print("\nEmbedding & Upload 시작\n")
    process_jsonl_folder(folder)
    print("모두 완료되었습니다!")
