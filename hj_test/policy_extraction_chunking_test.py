import re
import json

# 1) 텍스트 로드
with open("policy_only.txt", encoding="utf-8") as f:
    text = f.read()

# 2) 관별 분리
chapters = re.split(r'\n제(?=\d+관)', text)[1:]

all_chunks = []
for chap in chapters:
    # “제1관” 중 숫자 부분만 추출
    chap_match = re.match(r'(\d+)관', chap)
    chapter = f"제{chap_match.group(1)}관" if chap_match else ""

    # 3) 조별 분리
    articles = re.split(r'\n제(?=\d+조)', chap)[1:]
    for art in articles:
        # “제2조” 중 숫자 부분만 추출
        art_match = re.match(r'(\d+)조', art)
        article = f"제{art_match.group(1)}조" if art_match else ""
        body = art.strip()

        # 4) 항·호별 분리
        clauses = re.split(r'(?=\n?①)', body)
        for clause in clauses:
            clause = clause.strip()
            if not clause:
                continue
            m = re.match(r'(①|②|③|④|⑤|⑥|⑦|⑧|⑨|⑩)(.*)', clause, re.S)
            clause_no, clause_text = (m.group(1), m.group(2).strip()) if m else ("", clause)

            # 5) JSON 오브젝트에 담기
            all_chunks.append({
                "chapter": chapter,    # e.g. "제1관"
                "article": article,    # e.g. "제1조"
                "clause": clause_no,   # e.g. "①"
                "text": clause_text
            })

# 6) JSON Lines 형식으로 저장
with open("chunks_clauses.jsonl", "w", encoding="utf-8") as out:
    for chunk in all_chunks:
        out.write(json.dumps(chunk, ensure_ascii=False) + "\n")

print(f"총 {len(all_chunks)}개의 (관-조-항) 청크를 chunks_clauses.jsonl에 저장했습니다.")