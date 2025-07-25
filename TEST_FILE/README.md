# RAG Query 프로젝트

이 저장소는 보험 약관 문서와 상품명을 Qdrant 벡터 데이터베이스에 업로드하고, LLM을 통해 질문에 답변을 생성하는 RAG(Retrieval-Augmented Generation) 파이프라인을 제공합니다.

---

## 1. 필수 패키지 설치

다음 명령어로 필요한 Python 패키지를 설치하세요:

```bash
pip install requests PyPDF2 pdfplumber tiktoken langchain-text-splitters qdrant-client tqdm python-dotenv openai
```

- **requests**: HTTP 요청
- **PyPDF2**: PDF 읽기·쓰기
- **pdfplumber**: PDF → 텍스트 추출
- **tiktoken**: 토큰 수 계산 및 분할
- **langchain-text-splitters**: 텍스트 청크 분할
- **qdrant-client**: Qdrant 연동
- **tqdm**: 진행 상태 표시
- **python-dotenv**: `.env` 파일 로드
- **openai**: OpenAI API 연동 (선택)

---

## 2. Docker 기반 Qdrant 실행

1. 현재 실행 중인 컨테이너 확인:
   ```bash
   ```

docker ps

````
2. 기존 Qdrant 컨테이너 중지 및 삭제:
   ```bash
docker stop qdrant
docker rm qdrant
````

3. Qdrant 최신 이미지로 실행:
   ```bash
   ```

docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant\:latest

````

또는 `docker-compose up -d`를 사용할 수 있습니다.

---

## 3. 환경 변수 설정 (`.env`)

```dotenv
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=insurance_policies
INSURANCE_PRODUCTS_COLLECTION=insurance_products
EMBEDDING_API_URL=http://localhost:8001/embed
EMBEDDING_MODEL=intfloat/multilingual-e5-large-instruct
LLM_API_URL=http://localhost:8000/v1/chat/completions
LLM_MODEL_ID=/model/gen/Midm
TOP_K=5
````

- `QDRANT_COLLECTION`: 약관 청크 전용 컬렉션
- `INSURANCE_PRODUCTS_COLLECTION`: 상품명 전용 컬렉션

---

## 4. 실행 순서

1. **PDF → JSONL 변환**

   ```bash
   python batch_process_policies_rag.py ./pdf_dir
   ```

   - 결과: `./pdf_dir/jsonl/`에 JSONL 파일 생성

2. **JSONL → Qdrant 업로드 (약관)**

   ```bash
   python embed_upload_qdrant_api.py ./pdf_dir/jsonl
   ```
   - 컬렉션: `insurance_policies` (기존 삭제 후 재생성)

3. **상품명 JSON → Qdrant 업로드 (상품명)**

   ```bash
   python embed_upload_qdrant_api.py insurance_products.json
   ```

   - 컬렉션: `insurance_products` (기존 삭제 후 재생성)

4. **질의 응답 실행**

   ```bash
   python rag_query.py
   ```

   - 상품명 검색 → 약관 검색 → LLM 응답 스트리밍

---

## 5. 스크립트 요약

- **batch\_process\_policies\_rag.py**: PDF에서 조문별 청크 생성 → JSONL 저장
- **embed\_upload\_qdrant\_api.py**: JSONL/JSON → 임베딩 → Qdrant 업로드
- **rag\_query.py**: 질문 처리 → Qdrant 상품명 검색 → Qdrant 약관 검색 → LLM 응답

---

## 6. 참고

- `langchain-text-splitters`의 `RecursiveCharacterTextSplitter`를 사용하여 문맥을 최대한 유지하며 청크 분할
- `tiktoken`으로 모델별 토큰 수 측정 가능

---

