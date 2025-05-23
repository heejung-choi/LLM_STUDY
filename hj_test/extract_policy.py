import re
import os
from PyPDF2 import PdfReader, PdfWriter

# 1) 원본 PDF 파일 경로
infile = "20250401_HighFive그랑에이지변액연금보험무배당2504 (1).pdf"
reader = PdfReader(os.path.join(os.path.dirname(__file__), infile))

# 2) “제1관”이 나오는 첫 페이지 인덱스 자동 검색
start_page = None
for idx, page in enumerate(reader.pages):
    text = page.extract_text() or ""
    if "제1관" in text:
        start_page = idx
        break
if start_page is None:
    raise RuntimeError("PDF에서 '제1관'을 찾을 수 없습니다.")

# 3) 최대 '제n조(' 번호 검색 (start_page 이후 5페이지 이내)
max_article = 0
article_pattern = re.compile(r"제(\d+)조\(")
# start_page부터 5페이지(인덱스) 범위 내에서만 검색
end_search = min(start_page + 5, len(reader.pages))
for idx in range(start_page, end_search):
    text = reader.pages[idx].extract_text() or ""
    nums = [int(n) for n in article_pattern.findall(text)]
    if nums:
        max_article = max(max_article, max(nums))

# 4) PDF 추출 및 저장 (각종 제외 조건 적용)
writer = PdfWriter()
print(f"제{max_article}조")
out_path = "policy_only.pdf"
for idx in range(start_page, len(reader.pages)):
    page = reader.pages[idx]
    text = page.extract_text() or ""

    # 시작 전 페이지는 건너뜁니다
    if idx < start_page:
        continue
    # '제도성' 문구 포함 시 건너뜁니다
    if "제도성" in text:
        continue
    # '지정대리청구서비스특약 약관' 등장 시 종료합니다
    if "지정대리청구서비스특약 약관" in text:
        break
    # '제1관' 이후 5페이지 이상 지났고, 최대 조 번호 등장시(제{max_article}조() 형태) 종료합니다
    if idx - start_page >= 5 and f"제{max_article}조(" in text:
        break
    writer.add_page(page)

# 5) 결과 저장
with open(out_path, "wb") as f:
    writer.write(f)

print(f"policy_only.pdf 생성 완료: 페이지 {start_page+1}부터 제외 조건 전까지, 총 {len(writer.pages)}페이지)")
