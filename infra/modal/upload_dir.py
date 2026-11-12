"""Upload a local directory to the Modal Volume in chunks, with retries, skipping files already present.

`modal volume put` uploads a large directory in one go. On 13 Nov a 7.1 GB upload died after ~3 min with a
connection error and left nothing behind. This uploads ~40 files per batch and retries each batch.

Usage: MODAL_PROFILE=muhammadfaheem-mansha python infra/modal/upload_dir.py <local_dir> <remote_dir>
"""

import sys
import time
from pathlib import Path

import modal

local, remote = Path(sys.argv[1]), sys.argv[2].rstrip("/")
vol = modal.Volume.from_name("tem-axon-myelin")
try:
    have = {e.path.lstrip("/"): e.size for e in vol.listdir(remote, recursive=True)}
except Exception:  # noqa: BLE001 (remote dir does not exist yet)
    have = {}
files = sorted(p for p in local.rglob("*") if p.is_file())
todo = [p for p in files if have.get(f"{remote}/{p.relative_to(local)}".lstrip("/")) != p.stat().st_size]
print(f"{len(files)} files, {len(files) - len(todo)} already uploaded, {len(todo)} to go", flush=True)
for i in range(0, len(todo), 40):
    chunk = todo[i:i + 40]
    for attempt in range(5):
        try:
            with vol.batch_upload(force=True) as b:
                for p in chunk:
                    b.put_file(str(p), f"{remote}/{p.relative_to(local)}")
            break
        except Exception as e:  # noqa: BLE001
            print(f"chunk {i}: attempt {attempt + 1} failed: {e}", flush=True)
            time.sleep(10 * (attempt + 1))
    else:
        sys.exit(f"chunk {i} failed 5 times")
    print(f"{min(i + 40, len(todo))}/{len(todo)}", flush=True)
print("done", flush=True)
