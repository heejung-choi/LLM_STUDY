# batch_process_policies_rag.py

"""
batch_process_policies_rag.py

보험 약관 PDF → 페이지 단위 전체 청킹(Chunker) → JSONL 생성
Usage:
    python batch_process_policies_rag.py <PDF_DIR>
"""
import os
import sys
import json
import unicodedata
import argparse
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from PyPDF2 import PdfReader
import pdfplumber

# infrastructure/chunker.py에 정의된 청커
from infrastructure.chunker import RegexPolicyChunker  
from domain.models import PolicyChunk

# ── 환경변수 로드 ─────────────────────────────────────────
load_dotenv()
QDRANT_HOST               = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT               = int(os.getenv("QDRANT_PORT", "6333"))
INSURANCE_META_COLLECTION = os.getenv("INSURANCE_PRODUCTS_COLLECTION", "insurance_products")

# ── SALE_START_DT 매핑 로드 ─────────────────────────────────
def load_sale_start_map() -> dict[str, str]:
    client = QdrantClient(url=f"http://{QDRANT_HOST}:{QDRANT_PORT}", prefer_grpc=False)
    sale_map: dict[str, str] = {}
    offset, batch_size = 0, 100
    while True:
        points = client.scroll(
            collection_name=INSURANCE_META_COLLECTION,
            offset=offset,
            limit=batch_size,
            with_payload=True
        )
        if not points:
            break
        for pt in points:
            payload = getattr(pt, 'payload', pt if isinstance(pt, dict) else {}) or {}
            pid = payload.get("name") or payload.get("product_id")
            dt  = payload.get("start") or payload.get("sale_start_dt")
            if pid and dt and (pid not in sale_map or dt > sale_map[pid]):
                sale_map[pid] = dt  # 최신 판매 시작일 보관
        offset += len(points)
        if len(points) < batch_size:
            break
    return sale_map

SALE_MAP = load_sale_start_map()

# ── 텍스트 정제 함수 ─────────────────────────────────────────
def clean_text(text: str) -> str:
    t = unicodedata.normalize('NFC', text or '')
    return t.replace('\r', '\n').strip()

# ── PDF 디렉터리 일괄 처리 ─────────────────────────────────
def process_pdf_dir(pdf_dir: str):
    out_dir = os.path.join(pdf_dir, "jsonl_output")
    os.makedirs(out_dir, exist_ok=True)

    pdf_paths = [
        os.path.join(root, f)
        for root, _, files in os.walk(pdf_dir)
        for f in files if f.lower().endswith(".pdf")
    ]
    if not pdf_paths:
        print("ERROR: PDF 디렉터리에 PDF 파일이 없습니다.")
        sys.exit(1)

    # 청커 인스턴스 생성
    chunker = RegexPolicyChunker()

    for path in pdf_paths:
        product_id = os.path.splitext(os.path.basename(path))[0]
        print(f"▶ 처리 시작: {product_id}")

        # 페이지별 원문 텍스트 추출
        raw_pages = []
        reader = PdfReader(path)
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(reader.pages, start=1):
                txt = page.extract_text() or ""
                if not txt.strip():
                    txt = pdf.pages[i-1].extract_text() or ""
                raw_pages.append((i, clean_text(txt)))

        # sale_start_dt는 메타에서 이후 처리 단계에서 병합
        # 청커로 청크 생성
        chunks = chunker.chunk(
            raw_pages,
            product_id=product_id
        )

        # JSONL로 저장
        out_file = os.path.join(out_dir, f"{product_id}.jsonl")
        with open(out_file, "w", encoding="utf-8") as wf:
            for c in chunks:
                wf.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")
        print(f"[저장] {out_file} ({len(chunks)}개 청크)")

# ── 엔트리포인트 ─────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF → JSONL 청킹 스크립트")
    parser.add_argument("pdf_dir", help="PDF 원본 디렉터리 경로")
    args = parser.parse_args()
    process_pdf_dir(os.path.abspath(args.pdf_dir))
