import re
import os
import argparse
from PyPDF2 import PdfReader, PdfWriter


def process_pdf(in_path: str, out_dir: str):
    """
    PDF 파일(in_path)에서 '제1관'이 나타나는 페이지부터
    정의된 종료 조건 전까지의 페이지를 추출하여
    '<원본파일명>_policy_only.pdf' 형식으로 out_dir에 저장합니다.
    이미 '_policy_only.pdf'로 끝나는 파일은 처리하지 않습니다.
    """
    basename = os.path.splitext(os.path.basename(in_path))[0]
    # '_policy_only' 접미사가 있는 파일은 건너뜁니다
    if basename.lower().endswith("_policy_only"):
        print(f"'{in_path}'은 이미 처리된 파일이므로 건너뜁니다.")
        return

    reader = PdfReader(in_path)

    # 1) '제1관'이 등장하는 시작 페이지 찾기
    start_page = None
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if "제1관" in text:
            start_page = idx
            break
    if start_page is None:
        print(f"'{in_path}': '제1관'을 찾지 못해 건너뜁니다.")
        return

    # 2) 시작 페이지 이후 5페이지 내 최댓값 조번호 검색
    article_pattern = re.compile(r"제(\d+)조\(")
    max_article = 0
    for idx in range(start_page, min(start_page + 5, len(reader.pages))):
        nums = [int(n) for n in article_pattern.findall(reader.pages[idx].extract_text() or "")]
        if nums:
            max_article = max(max_article, max(nums))

    # 3) 페이지 추출 및 종료/제외 규칙
    writer = PdfWriter()
    for idx in range(start_page, len(reader.pages)):
        text = reader.pages[idx].extract_text() or ""
        if "제도성" in text:  # '제도성' 포함 시 제외
            continue
        if "지정대리청구서비스특약 약관" in text:  # 특약 제목 나오면 종료
            break
        if idx - start_page >= 5 and f"제{max_article}조(" in text:  # 최대 조 이후 발견 시 종료
            writer.add_page(reader.pages[idx])  # 여기서 페이지를 추가하고
            break                             # 루프 종료
        writer.add_page(reader.pages[idx])

    # 4) 출력 파일명 생성
    out_filename = f"{basename}_policy_only.pdf"
    out_path = os.path.join(out_dir, out_filename)

    # 5) 디렉터리 생성, 파일 쓰기
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, 'wb') as f:
        writer.write(f)

    print(f"{out_path} 생성 완료: {start_page+1}페이지부터 제외 조건 전까지, 총 {len(writer.pages)}페이지")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="원본 PDF 디렉터리에서 policy_only 섹션을 추출하여 별도 폴더에 저장합니다."
    )
    parser.add_argument(
        "pdf_dir",
        nargs='?',            # 인자를 선택적으로 받음 (없으면 현재 디렉터리)
        default='.',
        help="처리할 PDF 파일들이 있는 디렉터리 경로 (기본: 현재 폴더)"
    )
    parser.add_argument(
        "--out_dir", "-o",
        default="./output",
        help="추출된 PDF를 저장할 디렉터리 경로 (기본: ./output)"
    )
    args = parser.parse_args()

    pdf_dir = os.path.abspath(args.pdf_dir)
    out_dir = os.path.abspath(args.out_dir)

    # 처리 대상 PDF: 확장자 '.pdf'이며, '_policy_only.pdf' 접미사 없는 파일
    pdf_files = [f for f in os.listdir(pdf_dir)
                 if f.lower().endswith('.pdf') and not f.lower().endswith('_policy_only.pdf')]

    if not pdf_files:
        print(f"'{pdf_dir}'에 처리할 PDF 파일이 없습니다.")
    for pdf in pdf_files:
        process_pdf(os.path.join(pdf_dir, pdf), out_dir)
