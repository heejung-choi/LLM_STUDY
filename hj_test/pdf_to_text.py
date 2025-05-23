import pdfplumber

infile = "policy_only.pdf"
outfile = "policy_only.txt"

with pdfplumber.open(infile) as pdf:
    text = "\n".join(page.extract_text() or "" for page in pdf.pages)

with open(outfile, "w", encoding="utf-8") as f:
    f.write(text)

print(f"{outfile} 생성 완료")