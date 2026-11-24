"""T-28 G14: build an anonymised code snapshot for double-blind review (no data, masks, predictions or checkpoints).

Copies the evaluator package, analysis scripts, trainer definitions, frozen splits, committed metric files and the
pre-registered decision rules into an output folder, rewrites the licence holder, and fails if any identifying string
remains. Publishing the folder (e.g. anonymous.4open.science) is a manual owner step.

    .venv/bin/python scripts/release/make_anon_release.py [--out docs/paper/anon_release]
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
INCLUDE = ["src", "scripts/run", "scripts/eda", "baselines", "configs", "tests", "results", "pyproject.toml",
           "requirements.txt"]
SKIP_SUFFIX = {".log", ".pyc", ".tif", ".png", ".npz", ".pth", ".b2nd", ".pkl"}
SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints"}
# identifying patterns (owner, accounts, local paths); extend if the check below reports a hit
BAD = re.compile(r"muhammad|faheem|gurkani|faizan|mansha|i220485|/Volumes/|/Users/|claude|anthropic|\bowner\b", re.I)
# neutral rewrites applied to every copied text file before the check
REWRITE = [(re.compile(r"Claude's", re.I), "a non-expert rater's"), (re.compile(r"\bClaude\b", re.I), "a non-expert rater"),
           (re.compile(r"\bowner\b", re.I), "project lead")]
PRE_REG = REPO / "docs/notes/07-plan.md"


def copy_tree(src: Path, dst: Path) -> int:
    n = 0
    for p in src.rglob("*") if src.is_dir() else [src]:
        if p.is_dir() or p.suffix in SKIP_SUFFIX or SKIP_DIRS & set(p.parts):
            continue
        rel = p.relative_to(REPO)
        q = dst / rel
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)
        n += 1
    return n


def preregistration(dst: Path) -> None:
    """Hypothesis table (§2) and the decision records §7.2-§7.8, scrubbed of owner quotes and account names."""
    t = PRE_REG.read_text()
    hyp = t[t.index("## 2. Hypotheses"): t.index("\n## ", t.index("## 2. Hypotheses") + 5)]
    rec = t[t.index("### 7.2"):]
    body = "# Pre-registered hypotheses and decision rules\n\nExtracted from the project's decision record; " \
           "git commit hashes and dates of each registration are listed in the paper (Appendix A).\n\n" + hyp + "\n" + rec
    body = re.sub(r"\(owner[^)]*\)", "(project lead)", body)
    body = re.sub(r"\"[^\"]{0,300}(owner|Sir)[^\"]{0,300}\"", "\"…\"", body)
    (dst / "PREREGISTRATION.md").write_text(body)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "docs/paper/anon_release"))
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    n = sum(copy_tree(REPO / x, out) for x in INCLUDE if (REPO / x).exists())
    (out / "LICENSE").write_text((REPO / "LICENSE").read_text().replace("Muhammad Faheem", "Anonymous authors"))
    preregistration(out)
    for p in out.rglob("*"):
        if p.is_file() and p.suffix in {".py", ".md", ".json", ".csv", ".txt", ".toml"}:
            s = p.read_text(errors="ignore")
            s2 = s
            for pat, rep in REWRITE:
                s2 = pat.sub(rep, s2)
            if s2 != s:
                p.write_text(s2)
    (out / "README.md").write_text("""# Paired-fiber evaluation for TEM axon-myelin segmentation (anonymous review copy)

Code accompanying the submission "Selection, not pairing". No images, masks, predictions or model weights are
included: the challenge data licence is unstated, and the public datasets (ASTIH TEM1/TEM2 on DANDI, AxonPath on
Zenodo) must be downloaded from their sources.

- `src/axon_myelin_seg/evaluation/`: paired-fiber evaluator (official metrics, diagnostics, taxonomy, oracles,
  cluster bootstrap); tests in `tests/`.
- `src/axon_myelin_seg/postprocessing/pairing.py`: the shared pairing rule (D-11) and the frozen H5c rule.
- `baselines/`: nnU-Net trainers (incl. soft-weight trainer), AxonPath and Cellpose-SAM zero-shot runners.
- `scripts/run/`: every analysis in the paper (day*/phase*/t24_*/paper_*.py); each file's docstring gives its command.
- `configs/splits/`: frozen splits and overlap groups.
- `results/`: every metric table and JSON behind the paper's numbers.
- `PREREGISTRATION.md`: the hypotheses and decision rules as registered.

Install: `pip install -e .` (Python 3.11); nnU-Net v2, Cellpose and AxonPath from their own repositories.
""")
    hits = []
    for p in out.rglob("*"):
        if p.is_file():
            try:
                s = p.read_text(errors="ignore")
            except OSError:
                continue
            for m in BAD.finditer(s):
                hits.append(f"{p.relative_to(out)}: ...{s[max(0, m.start() - 30): m.end() + 30]!r}...")
    print(f"{n} files copied to {out}")
    if hits:
        print(f"IDENTIFYING STRINGS FOUND ({len(hits)}):")
        print("\n".join(hits[:40]))
        raise SystemExit(1)
    print("anonymity check passed")


if __name__ == "__main__":
    main()
