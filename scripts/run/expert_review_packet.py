"""Blind expert-review packet: T-20 (are B1's 'unlabelled_real' predictions real fibers?) + the Phase 0b
completeness-audit candidates (are unlabelled ring candidates real fibers?), rated together.

Items (seed 0):
- T-20: all B1 dev-val predictions classed `unlabelled_real` (61) + 40 random `other_bg`; outline = the prediction.
- Audit: the gallery samples of `phase0b_completeness_audit.py` (ours 48, TEM1 48, TEM2_dandi 48, AxonPath 15);
  outline = the ring candidate.
Every crop shows the same physical field of view (1.8 µm) at 256 px, raw image + one yellow outline, no GT tint,
so the rater sees neither the dataset nor the class. Order is shuffled.

Output (git-ignored, contains image crops; never upload — challenge data licence is unstated):
  data/processed/expert_review/index.html   rating page (5 categories, autosave in the browser, CSV export)
  data/processed/expert_review/img/*.png
  data/processed/expert_review/key.csv      item -> part, dataset, class, image, position (rater must not open)

    .venv/bin/python scripts/run/expert_review_packet.py
"""
from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from axon_myelin_seg.data.io import load_sample, read_mask
from axon_myelin_seg.evaluation.runner import score_image
from axon_myelin_seg.evaluation.taxonomy import ring_candidates

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data/processed/expert_review"
VIS = REPO / "data/processed/phase0b"
FOV_NM, SIZE = 1800, 256

spec = importlib.util.spec_from_file_location("audit", REPO / "scripts/run/phase0b_completeness_audit.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def crop(img: np.ndarray, mask: np.ndarray, cy: int, cx: int, px_nm: float) -> np.ndarray:
    half = round(FOV_NM / 2 / px_nm)
    pad = np.pad(img, half, mode="constant", constant_values=255)
    pm = np.pad(mask.astype(np.uint8), half)
    c = pad[cy: cy + 2 * half, cx: cx + 2 * half]
    m = pm[cy: cy + 2 * half, cx: cx + 2 * half]
    c = cv2.resize(c, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    m = cv2.resize(m, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    rgb = cv2.cvtColor(c, cv2.COLOR_GRAY2BGR)
    cnts, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(rgb, cnts, -1, (0, 220, 255), 1)  # yellow in BGR
    return rgb


def t20_items(rng: random.Random) -> list[dict]:
    names = json.loads((REPO / "configs/splits/dev.json").read_text())["val"]
    found = {"unlabelled_real": [], "other_bg": []}
    for n in names:
        pred = read_mask(REPO / "data/predictions/b1" / f"{n}.tif")
        img, gt = load_sample(n)
        r = score_image(pred, gt, image=img, return_classes=True)
        for k, c in r["_classes"]["pred"].items():
            if c in found:
                found[c].append((n, k))
    pick = [("unlabelled_real", x) for x in found["unlabelled_real"]]
    pick += [("other_bg", x) for x in rng.sample(found["other_bg"], min(40, len(found["other_bg"])))]
    items, cache = [], {}
    for cls, (n, k) in sorted(pick, key=lambda t: t[1][0]):
        if n not in cache:
            cache = {n: (load_sample(n)[0], read_mask(REPO / "data/predictions/b1" / f"{n}.tif"))}
        img, pred = cache[n]
        f = (pred == 2 * k) | (pred == 2 * k + 1)
        ys, xs = np.nonzero(f)
        cy, cx = int(ys.mean()), int(xs.mean())
        items.append({"part": "T-20", "dataset": "ours_B1_dev", "cls": cls, "image": n, "obj": int(k),
                      "cy": cy, "cx": cx, "png": crop(img, f, cy, cx, 2.152),
                      "n_found": len(found[cls])})
    return items


def audit_items() -> list[dict]:
    items = []
    for name in ["ours", "TEM1", "TEM2_dandi", "AxonPath"]:
        factory, px_nm, ds = audit.DATASETS[name]
        idx = pd.read_csv(VIS / f"audit_{name}_index.csv")
        want = set(idx.image)
        for img_name, _, img, gt in factory():
            if img_name not in want:
                continue
            cand = ring_candidates(img, gt, ds=ds, px_nm=px_nm)
            for r in idx[idx.image == img_name].itertuples():
                lab = cand[r.cy, r.cx]
                m = cand == lab if lab > 0 else np.zeros_like(cand, bool)
                items.append({"part": "audit", "dataset": name, "cls": "ring_candidate", "image": img_name,
                              "obj": int(lab), "cy": int(r.cy), "cx": int(r.cx),
                              "png": crop(img, m, int(r.cy), int(r.cx), px_nm)})
    return items


HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Fiber review</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#fafafa;--fg:#111;--card:#fff;--line:#ddd;--acc:#2563eb}
@media (prefers-color-scheme:dark){:root{--bg:#111;--fg:#eee;--card:#1c1c1c;--line:#333;--acc:#60a5fa}}
body{background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif;margin:0 auto;max-width:760px;padding:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:16px}
img{width:100%;max-width:512px;image-rendering:auto;display:block;margin:8px auto;border-radius:4px}
.opts{display:grid;gap:6px;margin-top:8px}
button{font:inherit;padding:8px 10px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg);text-align:left;cursor:pointer}
button.sel{border-color:var(--acc);outline:2px solid var(--acc)}
.nav{display:flex;gap:8px;justify-content:space-between;margin-top:12px;flex-wrap:wrap}
small{opacity:.75}
</style></head><body>
<h2>Myelinated-fiber review</h2>
<p><small>Each image shows ~1.8 µm of mouse white matter (TEM). Judge only the structure inside the <b>yellow outline</b>.
Keys 1–5 choose, ← → move. Your answers save in this browser; press <b>Export CSV</b> when done and send the file.</small></p>
<p><label>Rater name/initials: <input id="rater" style="font:inherit"></label></p>
<div class="card"><div id="pos"></div><img id="im" alt="crop"><div class="opts" id="opts"></div>
<p><label>Comment (optional): <input id="cm" style="font:inherit;width:100%"></label></p>
<div class="nav"><button id="prev">← Prev</button><button id="next">Next →</button><button id="exp">Export CSV</button></div></div>
<script>
const N=__N__, OPTS=__OPTS__;
let S={}; try{S=JSON.parse(localStorage.getItem('fr')||'{}')}catch(e){}
let i=0; const save=()=>{try{localStorage.setItem('fr',JSON.stringify(S))}catch(e){}};
const rater=document.getElementById('rater'); rater.value=S._rater||''; rater.oninput=()=>{S._rater=rater.value;save()};
function show(){const id=String(i+1).padStart(3,'0');document.getElementById('im').src='img/'+id+'.png';
 const done=Object.keys(S).filter(k=>k!=='_rater'&&S[k].c).length;
 document.getElementById('pos').textContent='Item '+(i+1)+' of '+N+' — rated '+done;
 const o=document.getElementById('opts');o.innerHTML='';
 OPTS.forEach((t,j)=>{const b=document.createElement('button');b.textContent=(j+1)+'. '+t;
  if(S[id]&&S[id].c==j+1)b.className='sel';b.onclick=()=>{S[id]={...(S[id]||{}),c:j+1};save();show()};o.appendChild(b)});
 const cm=document.getElementById('cm');cm.value=(S[id]&&S[id].m)||'';cm.oninput=()=>{S[id]={...(S[id]||{}),m:cm.value};save()}}
document.getElementById('prev').onclick=()=>{i=Math.max(0,i-1);show()};
document.getElementById('next').onclick=()=>{i=Math.min(N-1,i+1);show()};
document.onkeydown=e=>{if(e.target.tagName==='INPUT')return;if(e.key==='ArrowRight')document.getElementById('next').click();
 else if(e.key==='ArrowLeft')document.getElementById('prev').click();else if(e.key>='1'&&e.key<=String(OPTS.length)){
 const id=String(i+1).padStart(3,'0');S[id]={...(S[id]||{}),c:+e.key};save();show()}};
document.getElementById('exp').onclick=()=>{let csv='item,category,label,comment,rater\\n';
 for(let k=1;k<=N;k++){const id=String(k).padStart(3,'0');const r=S[id]||{};
  csv+=id+','+(r.c||'')+','+(r.c?JSON.stringify(OPTS[r.c-1]):'')+','+JSON.stringify(r.m||'')+','+JSON.stringify(S._rater||'')+'\\n'}
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));
 a.download='fiber_review_'+(S._rater||'rater')+'.csv';a.click()};
show();
</script></body></html>"""

OPTIONS = ["Real myelinated fiber — normal-looking",
           "Real myelinated fiber — abnormal (thin, redundant/onion, degenerating, oblique)",
           "Not a fiber — gap between fibers, organelle, debris, artefact",
           "Unmyelinated axon (no compact myelin)",
           "Can't tell"]


def main() -> None:
    rng = random.Random(0)
    items = t20_items(rng) + audit_items()
    rng.shuffle(items)
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    keys = []
    for j, it in enumerate(items, start=1):
        iid = f"{j:03d}"
        cv2.imwrite(str(OUT / "img" / f"{iid}.png"), it.pop("png"))
        keys.append({"item": iid, **it})
    pd.DataFrame(keys).to_csv(OUT / "key.csv", index=False)
    (OUT / "index.html").write_text(HTML.replace("__N__", str(len(items))).replace("__OPTS__", json.dumps(OPTIONS)))
    df = pd.DataFrame(keys)
    print(len(df), "items"); print(df.groupby(["part", "dataset", "cls"]).size())


if __name__ == "__main__":
    main()
