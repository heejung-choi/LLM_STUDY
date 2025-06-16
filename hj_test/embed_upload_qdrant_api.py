"""
JSONL로 청크된 약관 데이터를 임베딩 API 호출 후 Qdrant 벡터 DB에 저장
- .env 파일에서 QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION, EMBEDDING_API_URL 자동 로드
- 사용법:
    python embed_upload_qdrant_api.py <jsonl 디렉터리 경로>
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

# .env 자동 로드
load_dotenv()

# 환경변수 로드
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "insurance_policies")
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL","http://211.170.189.184:8000/v1/embeddings/")

if not EMBEDDING_API_URL:
    print("ERROR: .env에 EMBEDDING_API_URL이 설정되지 않았습니다.")
    sys.exit(1)

# Qdrant Client 초기화
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Qdrant Collection 생성 (존재하지 않을 경우)
def create_collection_if_not_exists():
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    if QDRANT_COLLECTION not in collection_names:
        print(f"NOTICE: Collection '{QDRANT_COLLECTION}'이 존재하지 않아 새로 생성합니다.")
        client.recreate_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=rest.VectorParams(size=1024, distance=rest.Distance.COSINE),
        )
    else:
        print(f"OK: Collection '{QDRANT_COLLECTION}' 존재 확인됨.")

# 임베딩 API 호출 (batch_input = list of text)
def get_embeddings(batch_input):
    response = requests.post(EMBEDDING_API_URL, json={"input": batch_input})
    response.raise_for_status()
    data = response.json()
    embeddings = data["data"] if "data" in data else data["result"]  # 모델에 따라 다름 처리
    return embeddings

# Main 업로드 처리
def process_and_upload(jsonl_dir):
    jsonl_files = [
        os.path.join(jsonl_dir, f) for f in os.listdir(jsonl_dir) if f.lower().endswith(".jsonl")
    ]

    if not jsonl_files:
        print("ERROR: JSONL 파일이 존재하지 않습니다.")
        sys.exit(1)

    print(f"총 {len(jsonl_files)}개 JSONL 파일을 처리합니다.\n")

    for jsonl_path in jsonl_files:
        print(f"Processing: {os.path.basename(jsonl_path)}")

        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line.strip()) for line in f if line.strip()]

        print(f" → {len(lines)}개의 청크를 임베딩 후 업로드합니다.")

        batch_size = 16
        for i in tqdm(range(0, len(lines), batch_size), desc="임베딩 진행중"):
            batch = lines[i:i + batch_size]
            batch_texts = [item["text"] for item in batch]

            # 임베딩 호출
            try:
                embeddings = get_embeddings(batch_texts)
            except Exception as e:
                print(f"ERROR: 임베딩 API 호출 오류: {e}")
                continue

            # Qdrant 업로드 준비
            points = []
            for j, (item, emb) in enumerate(zip(batch, embeddings)):
                vector = emb["embedding"] if isinstance(emb, dict) and "embedding" in emb else emb
                payload = {
                    "id": item.get("id"),
                    "chapter": item.get("chapter"),
                    "article": item.get("article"),
                    "clause_index": item.get("clause_index"),
                    "source_file": os.path.basename(jsonl_path),
                }

                points.append(
                    rest.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload=payload,
                    )
                )

            # Qdrant 업로드
            client.upsert(
                collection_name=QDRANT_COLLECTION,
                points=points
            )

            time.sleep(0.1)  # 과부하 방지 (optional)

        print(f"OK: {os.path.basename(jsonl_path)} 처리 완료\n")

# 엔트리포인트
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python embed_upload_qdrant_api.py <jsonl 디렉터리 경로>")
        sys.exit(1)

    jsonl_dir = sys.argv[1]

    if not os.path.isdir(jsonl_dir):
        print(f"ERROR: 디렉터리 '{jsonl_dir}' 이(가) 존재하지 않습니다.")
        sys.exit(1)

    print("Qdrant 서버 연결중...")
    create_collection_if_not_exists()

    print(f"\nEmbedding and Upload started: {jsonl_dir}\n")
    process_and_upload(jsonl_dir)

    print("\nCompleted!")
