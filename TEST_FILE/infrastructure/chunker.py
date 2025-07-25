# infrastructure/chunker.py

import re
import unicodedata
from typing import List, Tuple
from domain.models import IPolicyChunker, PolicyChunk
from langchain_text_splitters import RecursiveCharacterTextSplitter

class RegexPolicyChunker(IPolicyChunker):
    """
    정규표현식 기반 약관 청킹기 구현

    - 목차 페이지(한글 '목차') 제외
    - 공통부(보험계약관련 법·규정, 보험용어 해설)와 상품별 고유부 분리
    - 전체 페이지 텍스트를 장(제n관) 단위로 정교하게 분리
    - 각 장을 조(제m조) 단위로, 각 조를 항(①~⑩) 단위로 세분화
    - 항 단위(조문) 내 텍스트를 RecursiveCharacterTextSplitter로 세부 청크 생성
    """
    # 공통부 식별용 패턴
    COMMON_SECTION_PATTERNS = [
        re.compile(r"보험\s*계약\s*관련\s*법\W*규정", re.IGNORECASE),
        re.compile(r"보험\s*용어\s*해설", re.IGNORECASE),
    ]
    # 목차 판단 패턴 ('목차')
    TOC_PATTERN = re.compile(r"목\s*차")

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        # RecursiveCharacterTextSplitter 초기화
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " "]
        )

    @staticmethod
    def clean(text: str) -> str:
        # 텍스트 정제: NFC 정규화, CR->LF 통일, 양쪽 공백 제거
        t = unicodedata.normalize("NFC", text or "")
        return t.replace("\r", "\n").strip()

    def chunk(
        self,
        raw_pages: List[Tuple[int, str]],
        product_id: str
    ) -> List[PolicyChunk]:
        """
        지정된 상품에 대해 페이지별 원문을 받아 조각(청크) 단위로 분리

        Args:
            raw_pages: List of (page_no, text) tuples
            product_id: 약관이 속한 상품 식별자

        Returns:
            List of PolicyChunk instances
        """
        chunks: List[PolicyChunk] = []

        # 1) 목차 페이지 제외
        pages = [(no, txt) for no, txt in raw_pages if not self.TOC_PATTERN.search(txt)]

        for page_no, page_text in pages:
            # 텍스트 클리닝
            text = self.clean(page_text)

            # 2) 공통부 여부 판단
            is_common = any(p.search(text) for p in self.COMMON_SECTION_PATTERNS)
            # 장 제목이 있는 페이지는 상품고유부로 처리
            if re.search(r"(?m)^제\s*\d+관", text):
                is_common = False

            # 3) 장 분리: '제n관' 앞에서 분할
            chapter_blocks = re.split(r'(?m)^(?=제\s*\d+관)', text)
            for block in chapter_blocks:
                block = block.strip()
                if not block:
                    continue
                chap_m = re.match(r'(?m)^제\s*(\d+)관', block)
                chapter = f"제{chap_m.group(1)}관" if chap_m else ""

                # 4) 조 분리: '제m조(' 앞에서 분할
                if re.search(r'(?m)^제\s*\d+조\(', block):
                    article_blocks = re.split(r'(?m)^(?=제\s*\d+조\([^)]*\))', block)
                else:
                    article_blocks = [block]

                for art in article_blocks:
                    art = art.strip()
                    if not art:
                        continue
                    art_m = re.match(r'(?m)^제\s*(\d+)조', art)
                    article = f"제{art_m.group(1)}조" if art_m else ""

                    # 5) 항 분리: '①'~'⑩' 앞에서 분할
                    clauses = re.split(r'(?<=\))(?=[①-⑩])', art)
                    for idx, clause in enumerate(clauses):
                        clause = clause.strip()
                        if not clause:
                            continue

                        # 순수 목차 조->주계약 제거
                        if re.match(r'(?m)^제\s*\d+조\s*[\s\S]*?주계약\s*[-–]\s*\d+\s*$', clause):
                            continue
                        # 순수 주계약-숫자만 있는 행 제거
                        if re.match(r'^주계약\s*[-–]?\s*\d+\s*$', clause):
                            continue
                        # 주계약-숫자 접두사 제거, 본문 유지
                        clause = re.sub(r'^(?:주계약\s*[-–]?\s*\d+\s*)', '', clause).strip()
                        if not clause:
                            continue

                        # 6) 서브 청크 생성
                        for sub_idx, sub in enumerate(self.splitter.split_text(clause)):
                            chunks.append(PolicyChunk(
                                product_id=product_id,
                                page_no=page_no,
                                chapter=chapter,
                                article=article,
                                clause_idx=sub_idx,
                                text=sub,
                                is_common=is_common
                            ))
        return chunks
