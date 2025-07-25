# domain/models.py

from abc import ABC, abstractmethod
from typing import List, Tuple

class PolicyChunk:
    """
    보험 약관 한 조각을 나타내는 도메인 모델

    Attributes:
        product_id (str):
            약관이 속한 상품의 고유 식별자(예: 상품 코드 또는 명칭).
        page_no (int):
            해당 청크가 위치한 PDF 페이지 번호.
        chapter (str):
            청크가 속한 장 정보(예: '제2관').
        article (str):
            청크가 속한 조 정보(예: '제15조').
        clause_idx (int):
            한 조(article) 내에서 순번을 나타내는 인덱스.
        text (str):
            청크로 분리된 실제 텍스트 내용.
        is_common (bool):
            공통부 여부 플래그(True: 법·규정 또는 용어 해설 등 공통부).
    """
    def __init__(
        self,
        product_id: str,
        page_no: int,
        chapter: str,
        article: str,
        clause_idx: int,
        text: str,
        is_common: bool
    ):
        self.product_id = product_id
        self.page_no = page_no
        self.chapter = chapter
        self.article = article
        self.clause_idx = clause_idx
        self.text = text
        self.is_common = is_common

class IPolicyChunker(ABC):
    """
    약관 텍스트를 청크 단위로 분리해 PolicyChunk 리스트를 반환하는 인터페이스
    """
    @abstractmethod
    def chunk(
        self,
        raw_pages: List[Tuple[int, str]],
        product_id: str
    ) -> List[PolicyChunk]:
        pass
