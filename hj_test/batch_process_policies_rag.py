#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
batch_process_policies_rag.py

보험 약관/법령 PDF → 관/조 기반 Split + 필요 시 Recursive Split → VectorDB용 최적화 JSONL 생성

Usage:
    python batch_process_policies_rag.py <PDF_디렉터리>
"""

import os
import re
import json
import argparse
import unicodedata
from PyPDF2 import PdfReader, PdfWriter
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ——————————————————————————————————————————
# 설정
# ——————————————————————————————————————————
MAX_CHUNK_SIZE = 1024
CHUNK_OVERLAP = 50

# ——————————————————————————————————————————
# 텍스트 정제 함수
# ——————————————————————————————————————————
def clean_text(text: str) -> str:
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\s*\n\s*', '\n', text)
    text = text.replace('\uf0b7', '')
    return text

# ——————————————————————————————————————————
# policy-only PDF 추출
# ——————————————————————————————————————————
def extract_policy_only(input_pdf: str, output_pdf: str) -> bool:
    reader = PdfReader(input_pdf)
    start_page = None

    for idx, page in enumerate(reader.pages):
        if '제1관' in (page.extract_text() or ''):
            start_page = idx
            break
    if start_page is None:
        print(f"Skip {input_pdf}: '제1관' 시작 페이지 없음")
        return False

    article_pat = re.compile(r'제(\d+)조\(')
    max_article = 0
    for i in range(start_page, min(start_page + 5, len(reader.pages))):
        nums = article_pat.findall(reader.pages[i].extract_text() or '')
        if nums:
            max_article = max(max_article, max(map(int, nums)))

    writer = PdfWriter()
    for idx in range(start_page, len(reader.pages)):
        text = reader.pages[idx].extract_text() or ""

        if "제도성" in text:
            continue
        if "지정대리청구서비스특약 약관" in text:
            break
        if idx - start_page >= 5 and f"제{max_article}조(" in text:
            writer.add_page(reader.pages[idx])
            break
        writer.add_page(reader.pages[idx])

    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    with open(output_pdf, 'wb') as f:
        writer.write(f)
    print(f"[추출] {os.path.basename(output_pdf)}")
    return True

# ——————————————————————————————————————————
# PDF → 텍스트 변환
# ——————————————————————————————————————————
def pdf_to_text(input_pdf: str, output_txt: str):
    with pdfplumber.open(input_pdf) as pdf, \
         open(output_txt, 'w', encoding='utf-8') as out:
        for page in pdf.pages:
            out.write(page.extract_text() or '')
            out.write('\n')
    print(f"[텍스트변환] {os.path.basename(output_txt)}")

# ——————————————————————————————————————————
# 텍스트 → 관/조 기반 Split + Recursive Split 적용
# ——————————————————————————————————————————
def chunk_text(text: str) -> list:
    text = clean_text(text)
    chapters = re.split(r'\n(?=제\d+관)', text)
    if len(chapters) == 1 and not chapters[0].startswith('제'):
        chapters = ['제1관\n' + chapters[0]]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = []
    for chap in chapters:
        chap_match = re.match(r'제(\d+)관', chap)
        if not chap_match:
            continue
        chap_no = chap_match.group(1)

        # "제n조(타이틀)" 가 "새로운 줄의 처음에 등장할 때만 split"
        #articles = re.split(r'(?m)^\s*(?=제\d+조\()', chap)
        articles = re.split(r'(?m)^\s*(?=제\d+조\([^\)]*\)\s*$)', chap)
        
        for art in articles:
            art_match = re.match(r'제(\d+)조', art)
            if not art_match:
                continue
            art_no = art_match.group(1)
            art_text = art.strip()

            # — 목차/별표 제거 강화 —
            art_text = re.split(r'\n\(별표[^\n]*', art_text)[0].strip()

            if (
                re.fullmatch(r'제\d+조\([^\)]*\)\s*주계약[-\s]*\d*', art_text) or
                re.fullmatch(r'제\d+조\([^\)]*\)\s*－\d*－?', art_text) or
                len(art_text.splitlines()) <= 2 or
                re.fullmatch(r'제\d+조\([^\)]*\)\s*$', art_text)
            ):
                print(f"[목차제외] 제{art_no}조 (chapter {chap_no})")
                continue

            # Recursive Split 적용 (조문 제목은 첫 chunk에 반드시 포함)
            docs = text_splitter.create_documents([art_text])
            for idx, doc in enumerate(docs):
                chunks.append({
                    "id": f"{chap_no}-{art_no}-{idx}",
                    "chapter": f"제{chap_no}관",
                    "article": f"제{art_no}조",
                    "clause_index": idx,
                    "text": doc.page_content
                })

    print(f"[청킹] 총 {len(chunks)}개 청크 생성")
    return chunks

# ——————————————————————————————————————————
# 전체 디렉터리 일괄 처리
# ——————————————————————————————————————————
def process_all(pdf_dir: str):
    temp = os.path.join(pdf_dir, 'temp')
    out_jsonl = os.path.join(pdf_dir, 'jsonl')
    os.makedirs(temp, exist_ok=True)
    os.makedirs(out_jsonl, exist_ok=True)

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
        chunks = chunk_text(raw)

        with open(jsonl_file, 'w', encoding='utf-8') as jf:
            for c in chunks:
                jf.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"[저장] {os.path.basename(jsonl_file)}\n")

# ——————————————————————————————————————————
# 엔트리포인트
# ——————————————————————————————————————————
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='보험 약관/법령 PDF → 관/조 기반 Split + Recursive Split + RAG 최적화 JSONL 생성'
    )
    parser.add_argument('pdf_dir', help='원본 PDF가 위치한 디렉터리')
    args = parser.parse_args()
    process_all(os.path.abspath(args.pdf_dir))
