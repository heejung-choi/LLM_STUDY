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


# 실행순서

# v1
## 1.  policy_only.pdf 생성
-- python extract_policy.py    -> 초기 1개 버전

cd C:\Users\DGB생명\Documents\LLM_STUDY\vllm-test\hj_test
python extract_policy_multi.py pdf_dir --out_dir output

## 2. policy_only.txt 생성
python pdf_to_text.py         c   
## 3. 청킹테스트
python policy_extraction_chunking_test.py  

# v2
cd C:\Users\DGB생명\Documents\LLM_STUDY\vllm-test\hj_test
python extract_policy_multi.py pdf_dir --out_dir output
