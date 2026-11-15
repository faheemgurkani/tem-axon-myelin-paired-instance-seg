"""Insert or refresh a GitHub-style table of contents in a Markdown file.

The TOC sits between <!-- toc --> and <!-- /toc --> markers (added after the first paragraph block if absent).
Headings ## and ### are listed. Re-run after editing headings:  python3 scripts/toc.py PLAN.md docs/notes/07-plan.md
"""
import re
import sys


def slug(h: str) -> str:
    s = h.strip().lower()
    s = re.sub(r"[^\w\- ]", "", s)
    return s.replace(" ", "-")


def build(text: str) -> str:
    body = re.sub(r"<!-- toc -->.*?<!-- /toc -->\n?", "", text, flags=re.S)
    lines, seen, in_code = [], {}, False
    for ln in body.split("\n"):
        if ln.startswith("```"):
            in_code = not in_code
        if in_code:
            continue
        m = re.match(r"^(#{2,3}) (.+)$", ln)
        if not m:
            continue
        title = m.group(2).strip()
        base = slug(title)
        n = seen.get(base, 0)
        seen[base] = n + 1
        anchor = base if n == 0 else f"{base}-{n}"
        indent = "  " * (len(m.group(1)) - 2)
        lines.append(f"{indent}- [{title}](#{anchor})")
    toc = "<!-- toc -->\n**Contents**\n\n" + "\n".join(lines) + "\n<!-- /toc -->\n"
    # insert after the title block (first heading + following paragraph(s) up to the first '---' or '##')
    idx = re.search(r"\n(---\n|## )", body)
    pos = idx.start() + 1 if idx else len(body)
    return body[:pos] + toc + "\n" + body[pos:]


for path in sys.argv[1:]:
    t = open(path).read()
    open(path, "w").write(build(t))
    print("toc ->", path)
