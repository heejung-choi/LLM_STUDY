# 설치패키지
## requests
HTTP 요청을 위해 사용 (pip install requests)

## PyPDF2
PDF 읽기·쓰기용 (pip install PyPDF2)

## pdfplumber
PDF → 텍스트 추출용 (pip install pdfplumber)

## tiktoken
pip install tiktoken
OpenAI에서 제공하는 공식 토크나이저 라이브러리로, GPT-계열 모델이 내부적으로 사용하는 BPE(Byte Pair Encoding) 토크나이저를 그대로 제공
gpt-3.5-turbo, gpt-4o-mini 등 다양한 모델별 토큰화 방식을 지원
모델별로 미리 정의된 인코딩 설정(encoding_for_model("gpt-4o-mini"))을 바로 불러올 수 있음
빠른 토큰 수 계산
대량의 텍스트를 빠르게 토큰 단위로 분할하고, 토큰 수를 정확히 계산
RAG 파이프라인에서 “이 청크가 몇 토큰인지”를 즉시 파악해 분할 기준에 활용 가능

## langchain-text-splitters
pip install langchain-text-splitters
텍스트를 청크(조각)로 나누어주는 라이브러리

RecursiveCharacterTextSplitter:
단락 → 문장 → 단어 순서로 텍스트를 잘게 나눔 (가장 많이 씀)

CharacterTextSplitter:
문자 기준으로 단순히 나눔 (덜 똑똑함)

TokenTextSplitter:
토큰 기준으로 나눔 (tiktoken 사용, 정확한 토큰 기반 split)

RecursiveCharacterTextSplitter 특징
구분자 리스트를 줌 → 예) ["\n\n", "\n", " ", ""]
단락은 가능하면 함께 유지
안 되면 문장으로 → 그래도 안 되면 단어로 → 그래도 안 되면 문자로 → 재귀적으로 쪼갬.

즉, 문맥 유지 최대한 노력하면서 쪼갬 → GPT에 넣기 가장 좋은 방식.

# qdrant-client
pip install qdrant-client

# tqdm
pip install tqdm

# python-dotenv
pip install python-dotenv

# 실행순서

## v1 초기버전
### 1.  policy_only.pdf 생성
-- python extract_policy.py    -> 초기 1개 버전

cd C:\Users\DGB생명\Documents\LLM_STUDY\vllm-test\hj_test
python extract_policy_multi.py pdf_dir --out_dir output

### 2. policy_only.txt 생성
python pdf_to_text.py         c   
### 3. 청킹테스트
python policy_extraction_chunking_test.py  

## v2 수정버전
cd C:\Users\DGB생명\Documents\LLM_STUDY\vllm-test\hj_test
python extract_policy_multi.py pdf_dir --out_dir output

## V3 수정버전
python batch_process_policies.py pdf_dir

## V4 수정버전
python batch_process_policies_rag.py  pdf_dir


## qdant
### docker가 사용중인가?
docker ps 

### docker 실행방법
docker-compose up -d

### qdrant
# 1) 기존 컨테이너 중지
docker stop qdrant

# 2) 기존 컨테이너 삭제
docker rm qdrant

# 3) 새로 실행
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest

## 최종 실행 방법
1️⃣ PDF → JSONL 추출
python batch_process_policies_rag.py ./pdf_dir

# 결과가 ./pdf_dir/jsonl/ 아래 JSONL로 생성됨.

2️⃣ JSONL → 임베딩 API 호출 후 Qdrant에 업로드
python embed_upload_qdrant_api.py ./pdf_dir/jsonl

# 최종구조
[ batch_process_policies_rag.py ] → JSONL 청크 생성
          |
          v
[ embed_upload_qdrant_api.py ] → FastAPI 서버에 embedding 요청
          |
          v
[ FastAPI 서버 (embedding API) ] → E5 모델로 embedding
          |
          v
[ embed_upload_qdrant_api.py ] → Qdrant에 embedding 벡터 저장
          |
          v
[ Qdrant ] → Vector search 가능