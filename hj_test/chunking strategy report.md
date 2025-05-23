# sLLM-based Knowledge Search: Insurance Policy Chunking Strategy

## 1. 개요

보험 약관 문서를 sLLM 기반 지식 검색 시스템에 최적화하기 위해 적용된 **청킹(Chunking) 전략**을 작성하였습니다.
약관 텍스트를 의미 단위로 분할하여 검색 효율을 높이고, 벡터 인덱싱 및 RAG(검색-생성) 파이프라인에 적합한 형식으로 데이터를 가공하기 위한 가이드라인을 포함하였습니다.

청킹 단계(토큰 분할 전략) 뿐만 아니라, 전체 RAG 전처리 파이프라인(PDF→policy-only→텍스트→청킹→JSONL)에 대해 설계하였습니다.

---

## 2. 목적 및 배경

* **검색 정확도 제고**: 약관 조항은 문서 길이가 길어, 전체를 한 번에 LLM에 주입하면 관련 정보 추출이 어렵기 때문에 추출 가능한 형태로 분할 하였습니다.
* **성능 최적화**: 의미 단위로 분할된 청크는 벡터 DB에서 효율적으로 유사도 검색을 지원합니다.
* **모듈화와 재사용성**: 분할 기준과 메타데이터 스키마를 표준화하여 향후 다른 약관에도 적용 가능하도록 설계하였습니다.

---
## 3.약관 문서 구성 및 초기 검토

- **전체 섹션 구성**  
  1. 약관 이용 가이드북  
  2. 약관 요약서  
  3. 주요 민원사항  
  4. 보험용어 해설  
  5. 보험계약 관련 법·규정  
  6. 약관 규정(주계약)  
  7. 별표(상품별 차등사항: 보험금 지급 기준표·적립이율 계산)  
  8. 별표(공통사항: 장해분류표·재해분류표)  
  9. 특별약관  

- **상품별 변동 섹션**: 1, 2, 6, 7 (상품마다 내용·형식 상이)  
- **현재 진행 상황**  
  - 핵심 내용인 **6번 약관 규정(주계약)**만 우선 PDF로 발췌하여 청킹 테스트를 수행  
- **향후 계획**  
  - 나머지 섹션들(가이드북, 요약서, 별표 등)에 대해서도 각 섹션 특성에 맞는 최적의 청킹 전략을 설계·적용할 예정
---
## 4. 전처리 단계

1. **PDF 텍스트 추출**

   * `pdfplumber` 또는 `PyPDF2`를 사용하여 PDF에서 텍스트를 페이지 단위로 추출
   * 헤더·푸터, 목차, 불필요한 공백·특수 문자를 제거
   
2. **인코딩 및 개행 정규화**

   * UTF-8로 통일
   * 모든 개행(\n) 문자를 CRLF→LF로 통일

3. **초기 분할: 관 단위**

   * `\n제\d+관` 패턴으로 대분류(제1관\~제N관) 분리 

---

## 5. 청킹 단위 정의

| 단계      | 분할 기준                    | 설명                                  |
| ------- | ------------------------ | ----------------------------------- |
| **관**   | `제n관`                    | 대분류 단위. 각 관별로 독립 처리                 |
| **조**   | `제m조`                    | 중분류 단위. 문서의 주요 조항(Article) 분리       |
| **항·호** | `(1)`, `(2)`, `①`, `②` 등 | 소분류 단위. 세부 항목(Clause/Sub-clause) 분리 |

**분할 예시**

```text
제3관
제3조(보험금 지급)
① 보험금은…
② 다음의 경우…
```

위 예시는 `chapter=3`, `article=3`, `clause_index=0~1` 두 개의 청크로 분할하였습니다.

---

## 6. 청크 크기 및 오버랩(Overlap)

1. **최대 토큰 수**

   * 모델 최대 토큰의 70% 이내 목표 (예: 1,024 토큰 모델 → 약 720 토큰)
   * `tiktoken` 라이브러리를 이용해 실제 토크나이저 기준 토큰 수 계산 후 분할
2. **오버랩 적용**

   * 이웃 청크 간 50\~100 토큰 중복(슬라이딩 윈도우 방식)으로 문맥 흐름 유지
3. **긴 조 처리**

   * 단일 조가 토큰 한도를 초과할 경우, **고정 슬라이딩 윈도우**(예: 250단어 길이, 50단어 오버랩)로 세분화

---

## 7. 메타데이터 스키마

청크별 JSON 객체 구조:

```json
{
  "id": "3-3-0",          // chapter-article-clause 순번
  "chapter": "제3관",
  "article": "제3조",
  "clause_index": 0,       // 동일 조 내에서 순번
  "text": "보험금은 ...",
  "source": "policy_X.pdf"
}
```

| 필드             | 설명                 |
| -------------- | ------------------ |
| `id`           | 고유 식별자 ("관-조-항순번") |
| `chapter`      | 원문 장 단위 (`제n관`)    |
| `article`      | 원문 조 단위 (`제m조`)    |
| `clause_index` | 조 내 청크 순번(0부터)     |
| `text`         | 청크 본문 텍스트          |
| `source`       | 원본 PDF 파일명         |

---
## 8. python 설치 패키지
## requests
HTTP 요청을 위해 사용 (pip install requests)

## PyPDF2
PDF 읽기·쓰기용 (pip install PyPDF2)

## pdfplumber
PDF → 텍스트 추출용 (pip install pdfplumber)

## tiktoken
pip install tiktoken
- OpenAI에서 제공하는 공식 토크나이저 라이브러리로, GPT-계열 모델이 내부적으로 사용하는 BPE(Byte Pair Encoding) 토크나이저를 그대로 제공
- gpt-3.5-turbo, gpt-4o-mini 등 다양한 모델별 토큰화 방식을 지원
- 모델별로 미리 정의된 인코딩 설정(encoding_for_model("gpt-4o-mini"))을 바로 불러올 수 있음
- 빠른 토큰 수 계산
- 대량의 텍스트를 빠르게 토큰 단위로 분할하고, 토큰 수를 정확히 계산
- RAG 파이프라인에서 “이 청크가 몇 토큰인지”를 파악해 분할 기준에 활용 가능

## 9. 구현 예시 코드 (batch_process_policies.py 기반 상세)

아래는 `batch_process_policies.py` 파일의 주요 함수들을 중심으로 **청킹 파이프라인 전체**를 설명한 구현 예시입니다.

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
from PyPDF2 import PdfReader, PdfWriter
import pdfplumber
from tiktoken import encoding_for_model

# ——————————————————————————————————————————
# 전역 설정
# ——————————————————————————————————————————
ENC = encoding_for_model('gpt-4o-mini')  # GPT-4o-mini 모델 기준 토크나이저 로드
MAX_TOKENS = 720                         # 청크당 최대 토큰 수
OVERLAP_TOKENS = 50                      # 인접 청크 간 중복 토큰 수

# ——————————————————————————————————————————
# 1) 약관에서 '정책 전용(policy-only)' 페이지만 추출
# ——————————————————————————————————————————
def extract_policy_only(input_pdf: str, output_pdf: str) -> bool:
    """
    input_pdf에서 '제1관'이 시작되는 페이지부터
    특정 종료 조건 전까지 페이지를 뽑아 policy-only PDF 생성.
    종료 조건:
      1) '제도성' 포함 → 해당 페이지만 건너뛰기
      2) '지정대리청구서비스특약 약관' 발견 → 즉시 종료
      3) 시작 후 5페이지 이상 탐색, 최대 조번호 등장 → 해당 페이지 포함 후 종료
    """
    reader = PdfReader(input_pdf)
    start_page = None

    # — 시작 페이지(‘제1관’) 찾기
    for idx, page in enumerate(reader.pages):
        if '제1관' in (page.extract_text() or ''):
            start_page = idx
            break
    if start_page is None:
        print(f"Skip {input_pdf}: '제1관' 시작 페이지 없음")
        return False

    # — 시작 후 5페이지 내에 등장한 최대 조번호 파악
    article_pat = re.compile(r'제(\d+)조\(')
    max_article = 0
    for i in range(start_page, min(start_page + 5, len(reader.pages))):
        nums = article_pat.findall(reader.pages[i].extract_text() or '')
        if nums:
            max_article = max(max_article, max(map(int, nums)))

    writer = PdfWriter()
    # — 추출 규칙에 따라 페이지 순차 추가
    for idx in range(start_page, len(reader.pages)):
        text = reader.pages[idx].extract_text() or ""

        # (1) '제도성' 포함 시 건너뛰기
        if "제도성" in text:
            continue
        # (2) 특약 제목 등장 시 즉시 종료
        if "지정대리청구서비스특약 약관" in text:
            break
        # (3) 5페이지 이상 지났고 최대 조번호 조문 등장 시 해당 페이지만 추가 후 종료
        if idx - start_page >= 5 and f"제{max_article}조(" in text:
            writer.add_page(reader.pages[idx])
            break
        # (4) 그 외 모든 페이지 추가
        writer.add_page(reader.pages[idx])

    # — 출력 경로 생성 및 저장
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    with open(output_pdf, 'wb') as f:
        writer.write(f)
    print(f"[추출] {os.path.basename(output_pdf)}")
    return True

# ——————————————————————————————————————————
# 2) policy-only PDF → 텍스트 추출
# ——————————————————————————————————————————
def pdf_to_text(input_pdf: str, output_txt: str):
    """
    pdfplumber를 사용해 policy-only PDF 전체를 텍스트로 변환하여 저장.
    """
    with pdfplumber.open(input_pdf) as pdf, \
         open(output_txt, 'w', encoding='utf-8') as out:
        for page in pdf.pages:
            out.write(page.extract_text() or '')
            out.write('\n')
    print(f"[텍스트변환] {os.path.basename(output_txt)}")

# ——————————————————————————————————————————
# 3) 텍스트 → 의미 단위(관-조-항)별 슬라이딩 윈도우 청킹
# ——————————————————————————————————————————
def chunk_text(text: str, source: str) -> list:
    """
    1) '\n제n관' 기준으로 챕터 분리
    2) '제m조' 기준으로 조문 분리
    3) 각 조문 텍스트를 토큰화 → MAX_TOKENS 크기 슬라이딩 윈도우 방식으로 분할
    4) OVERLAP_TOKENS 만큼 겹치며 다음 청킹 시작
    5) 메타데이터(id, chapter, article, clause_index, source) 포함된 청크 리스트 반환
    """
    chunks = []
    # (1) 챕터 분리
    chapters = re.split(r'\n(?=제\d+관)', text)
    for chap in chapters:
        chap_match = re.match(r'제(\d+)관', chap)
        if not chap_match:
            continue
        chap_no = chap_match.group(1)

        # (2) 조문 분리
        articles = re.split(r'(?=제\d+조)', chap)
        for art in articles:
            art_match = re.match(r'제(\d+)조', art)
            if not art_match:
                continue
            art_no = art_match.group(1)

            art_text = art.strip()
            tokens = ENC.encode(art_text)
            total = len(tokens)

            # (3) 슬라이딩 윈도우로 분할
            start, idx = 0, 0
            while start < total:
                end = min(start + MAX_TOKENS, total)
                chunk_tok = tokens[start:end]
                chunk_txt = ENC.decode(chunk_tok).strip()

                # (4) 메타데이터와 함께 리스트에 추가
                chunks.append({
                    "id": f"{chap_no}-{art_no}-{idx}",
                    "chapter": f"제{chap_no}관",
                    "article": f"제{art_no}조",
                    "clause_index": idx,
                    "text": chunk_txt,
                    "source": source
                })
                idx += 1
                start += (MAX_TOKENS - OVERLAP_TOKENS)

    print(f"[청킹] 총 {len(chunks)}개 청크 생성")
    return chunks

# ——————————————————————————————————————————
# 4) 전체 디렉터리 재귀 탐색 후 일괄 처리
# ——————————————————————————————————————————
def process_all(pdf_dir: str):
    """
    1) pdf_dir 하위에서 .pdf 파일 재귀 수집 (temp/jsonl 폴더 제외)
    2) 각 PDF → extract_policy_only → pdf_to_text → chunk_text → JSONL 저장
    """
    temp = os.path.join(pdf_dir, 'temp')
    out_jsonl = os.path.join(pdf_dir, 'jsonl')
    os.makedirs(temp, exist_ok=True)
    os.makedirs(out_jsonl, exist_ok=True)

    # (1) 재귀 탐색하여 처리 대상 PDF 목록 생성
    pdfs = []
    for root, dirs, files in os.walk(pdf_dir):
        if os.path.basename(root).lower() in ('temp', 'jsonl'):
            continue
        for f in files:
            if f.lower().endswith('.pdf') and 'policy_only' not in f.lower():
                pdfs.append(os.path.join(root, f))
    if not pdfs:
        print("처리할 PDF 파일이 없습니다.")
        return

    # (2) 각 PDF 처리 루프
    for path in pdfs:
        base = os.path.splitext(os.path.basename(path))[0]
        policy_pdf = os.path.join(temp, f"{base}_policy_only.pdf")
        txt_file   = os.path.join(temp, f"{base}.txt")
        jsonl_file = os.path.join(out_jsonl, f"{base}.jsonl")

        print(f"▶ 처리 시작: {base}")
        if not extract_policy_only(path, policy_pdf):
            continue

        pdf_to_text(policy_pdf, txt_file)
        raw = open(txt_file, 'r', encoding='utf-8').read()
        chunks = chunk_text(raw, f"{base}.pdf")

        # (3) JSONL 파일로 청크 저장
        with open(jsonl_file, 'w', encoding='utf-8') as jf:
            for c in chunks:
                jf.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"[저장] {os.path.basename(jsonl_file)}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='여러 PDF 일괄 처리 → 청킹 JSONL 생성'
    )
    parser.add_argument('pdf_dir', help='원본 PDF가 위치한 디렉터리')
    args = parser.parse_args()
    process_all(os.path.abspath(args.pdf_dir))
```

## 10. 도구 및 의존성
* Python 3.9+
* `pdfplumber`, `PyPDF2` (PDF 추출)
* `tiktoken` (토큰 계산)
* `openai` SDK (임베딩 생성)
* `qdrant-client` (벡터 DB 연동)

## 11. 향후 추가로 해야할 사항들
1. **다양한 약관 케이스 테스트**
    - 현재 2개의 약관으로 json 파일 생성 테스트 했으나, 추후 전체 약관으로 테스트하며 전체 RAG 전처리 파이프라인(PDF→policy-only→텍스트→청킹→JSONL) 수정필요
    - 현재 약관의 규정부분 중 주계약 부분에 대해서만 전처리를 진행했으나, 별표 부분(보험금 지급 기준표, 장해분류표 등), 특별약관, 약관요약서 등에 대해서도 추가 필요 

2. **임베딩 생성 & 업서트**  
   - JSONL의 각 청크를 OpenAI 임베딩 API로 벡터화  
   - Qdrant 컬렉션에 벡터와 메타데이터 업서트  

3. **벡터 검색 & 리트리버 구성**  
   - 사용자 질문을 임베딩 → Qdrant에서 상위 k개 청크 검색  
   - 검색된 청크를 Prompt에 조합하여 LLM에 전달  

4. **캐싱·세션 관리**  
   - Redis를 이용해 쿼리/답변 캐싱, 대화 컨텍스트 유지  

5. **테스트·검증·배포 준비**  
   - 단위 테스트, 성능 측정, Docker 이미지화  

## 12. 약관 문서 구성 및 섹션별 청킹 전략 (임시 - 추후 변동 예정)

| 섹션 번호 | 섹션명                                                                          |
|----------|---------------------------------------------------------------------------------|
| 1        | 약관 이용 가이드북                                                               |
| 2        | 약관 요약서                                                                     |
| 3        | 주요 민원사항                                                                   |
| 4        | 보험용어 해설                                                                   |
| 5        | 보험계약 관련 법·규정                                                            |
| 6        | 약관 규정(주계약)                                                                |
| 7        | 별표(상품별 다른 사항: 보험금 지급 기준표, 적립이율 계산)                         |
| 8        | 별표(공통사항: 장해분류표, 재해분류표)                                           |
| 9        | 특별약관                                                                        |

---

### 섹션 1. 약관 이용 가이드북
- **분할 기준**  
  - 목차(챕터) → 소제목별 분리  
  - 소제목별 길이가 길면 문단 단위로 슬라이딩 윈도우(예: 500토큰, 오버랩 100토큰)  
- **메타데이터 예시**  
  ```json
  {
    "section": 1,
    "chapter": "가이드북 제2장",
    "subsection": "2.1 보험 가입 절차",
    "chunk_index": 0,
    "text": "…"
  }
  ```

### 섹션 2. 약관 요약서
- **분할 기준**  
  - 번호 매겨진 요약 블록별(Ⅰ, Ⅱ 등) 분리  
  - 각 블록이 길면 내부 문단별로 300~500토큰 단위로 분할  
- **메타데이터 예시**  
  ```json
  {
    "section": 2,
    "summary_item": "Ⅰ. 보장 내용",
    "chunk_index": 0,
    "text": "…"
  }
  ```

### 섹션 3. 주요 민원사항
- **분할 기준**  
  - 질문–답변(Q&A) 한 쌍을 하나의 청크  
  - 답변이 길면 400토큰 단위로 슬라이딩 분할  
- **메타데이터 예시**  
  ```json
  {
    "section": 3,
    "question": "보험료는 어떻게 납부하나요?",
    "chunk_index": 0,
    "text": "…"
  }
  ```

### 섹션 4. 보험용어 해설
- **분할 기준**  
  - 용어별: 용어명＋해설 본문을 하나의 청크  
  - 해설이 길면 200~300토큰 단위로 분할  
- **메타데이터 예시**  
  ```json
  {
    "section": 4,
    "term": "면책조건",
    "chunk_index": 0,
    "text": "…"
  }
  ```

### 섹션 5. 보험계약 관련 법·규정
- **분할 기준**  
  - 법령 조문별(제n조, 제n조의m) 분리  
  - 긴 조문은 슬라이딩 윈도우(720토큰, 50토큰 오버랩)로 세분화  
- **메타데이터 예시**  
  ```json
  {
    "section": 5,
    "law": "보험업법",
    "article": "제12조",
    "chunk_index": 0,
    "text": "…"
  }
  ```

### 섹션 6. 약관 규정(주계약)
- **분할 기준**  
  - (제n관→제m조→항·호 토큰 단위 슬라이딩 윈도우) 적용  -> 현재 적용 상태
- **메타데이터 예시**  
  ```json
  {
    "section": 6,
    "chapter": "제3관",
    "article": "제3조",
    "clause_index": 0,
    "text": "…"
  }
  ```

### 섹션 7. 별표(상품별 다른 사항)
- **분할 기준**  
  - 표 단위 분리  
  - 표가 크면 행(row) 단위로 추가 청킹  
- **메타데이터 예시**  
  ```json
  {
    "section": 7,
    "table": "보험금 지급 기준표",
    "row_index": 3,
    "text": "…"
  }
  ```

### 섹션 8. 별표(공통사항 표)
- **분할 기준**  
  - 각 표 전체를 하나의 청크  
  - 큰 표는 행 단위 슬라이딩 윈도우로 세분화  
- **메타데이터 예시**  
  ```json
  {
    "section": 8,
    "table": "장해분류표",
    "chunk_index": 0,
    "text": "…"
  }
  ```

### 섹션 9. 특별약관
- **분할 기준**  
  - 특약별로 분리(“암진단특약”, “6대질병면제특약” 등)  
  - 본문이 길면 500토큰 단위 슬라이딩 윈도우 분할  
- **메타데이터 예시**  
  ```json
  {
    "section": 9,
    "rider": "암진단특약",
    "chunk_index": 0,
    "text": "…"
  }
  ```