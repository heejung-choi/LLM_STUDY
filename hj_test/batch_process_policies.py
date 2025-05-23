#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
batch_process_policies.py

여러 약관 PDF 파일을 재귀적으로 일괄 처리하여:
1) 각 PDF에서 '제1관'~추출 정책 적용한 policy-only PDF 생성
2) policy-only PDF를 텍스트로 변환
3) tiktoken 기반 토큰 수로 청킹 (최대 720토큰, 50토큰 오버랩)
4) 청크별 메타데이터 포함 JSONL 저장
5) 불필요한 공백·특수문자 제거(clean_text) 적용

Usage:
    python batch_process_policies.py <PDF_디렉터리>
"""

import os
import re
import json
import argparse
import unicodedata
from PyPDF2 import PdfReader, PdfWriter
import pdfplumber
from tiktoken import encoding_for_model

# ——————————————————————————————————————————
# 청크 크기 및 오버랩(Overlap) 설정
# ——————————————————————————————————————————
ENC = encoding_for_model('gpt-4o-mini')  # GPT-4o-mini 모델 기준 토크나이저 로드
MAX_TOKENS = 720                         # 청크당 최대 토큰 수
OVERLAP_TOKENS = 50                      # 인접 청크 간 중복 토큰 수

def clean_text(text: str) -> str:
    """
    텍스트 정제:
      - Unicode NFC 정규화
      - 탭 및 여러 공백 → 단일 공백
      - 불필요한 줄바꿈 주변 공백 제거
      - 특정 특수문자(예: '\uf0b7') 제거
    """
    # 1) Unicode 정규화
    text = unicodedata.normalize('NFC', text)
    # 2) 탭 및 다중 공백 → 단일 공백
    text = re.sub(r'[ \t]+', ' ', text)
    # 3) 줄바꿈 앞뒤 공백 제거  -> 줄바꿈 앞뒤 공백 제거 및 CRLF/LF 통일
    text = re.sub(r'\s*\n\s*', '\n', text)
    # 4) 특정 특수문자 제거
    text = text.replace('\uf0b7', '')
    return text

# ——————————————————————————————————————————
# 1) 약관에서 '정책 전용(policy-only)' 페이지만 추출
# ——————————————————————————————————————————
def extract_policy_only(input_pdf: str, output_pdf: str) -> bool:
    """
    input_pdf에서 '제1관'이 시작되는 페이지부터
    특정 종료 조건 전까지 페이지를 뽑아 policy-only PDF 생성.
    종료 조건:
      1) '제도성' 포함 → 해당 페이지 제외
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
# 3) 텍스트 정제 후 → 의미 단위(관-조-항)별 슬라이딩 윈도우 청킹
# ——————————————————————————————————————————
def chunk_text(text: str, source: str) -> list:
    """
    1) clean_text로 불필요 문자 제거
    2) '\n제n관' 기준 챕터 분리
    3) '제m조' 기준 조문 분리
    4) 각 조문 텍스트를 토큰화 → MAX_TOKENS 크기 슬라이딩 윈도우로 분할
    5) OVERLAP_TOKENS 만큼 겹침 처리
    6) 메타데이터(id, chapter, article, clause_index, source) 포함된 청크 리스트 반환
    """
    # 1) 텍스트 정제
    text = clean_text(text)

    chunks = []
    # (2) 챕터 분리
    # 챕터 헤더가 붙어 있지 않은 경우, 첫 챕터로 간주
    if len(chapters) == 1 and not chapters[0].startswith('제'):
        chapters = ['제1관\n' + chapters[0]]
    for chap in chapters:
        chap_match = re.match(r'제(\d+)관', chap)
        if not chap_match:
            continue
        chap_no = chap_match.group(1)

        # (3) 조문 분리
        articles = re.split(r'(?=제\d+조)', chap)
        for art in articles:
            art_match = re.match(r'제(\d+)조', art)
            if not art_match:
                continue
            art_no = art_match.group(1)

            art_text = art.strip()
            tokens = ENC.encode(art_text)
            total = len(tokens)

            # (4) 슬라이딩 윈도우 분할
            start, idx = 0, 0
            while start < total:
                end = min(start + MAX_TOKENS, total)
                chunk_tok = tokens[start:end]
                chunk_txt = ENC.decode(chunk_tok).strip()

                # (6) 메타데이터와 함께 리스트에 추가
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

# ——————————————————————————————————————————
# 엔트리포인트
# ——————————————————————————————————————————
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='여러 PDF 일괄 처리 → 청킹 JSONL 생성'
    )
    parser.add_argument('pdf_dir', help='원본 PDF가 위치한 디렉터리')
    args = parser.parse_args()
    process_all(os.path.abspath(args.pdf_dir))
