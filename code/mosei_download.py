#!/usr/bin/env python3
"""Extract the selected CMU-MOSEI clips from the remote Raw.zip (see SPEC_mosei.md).

Only the members listed in subset.csv are read: each thread holds its own RemoteZip object and
streams one member at a time into clips/<video_id>/<clip_id>.mp4. A file is accepted when ffprobe
finds an audio and a video stream and the container duration is within 0.5 s of the subset `dur`.
Downloads are retried with exponential backoff. Re-running skips clips that are already valid.

Outputs (in /home/user/data/mosei/):
  clips/<video_id>/<clip_id>.mp4   validated clips
  download_status.csv              one row per subset clip: status, attempts, probe duration, error
  download_excluded.csv            clips that failed after all retries
  download.log                     progress log
"""
import argparse
import csv
import json
import os
import random
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import remotezip
import zipfile_deflate64  # noqa: F401  (registers the deflate64 decompressor for zipfile)

URL = "https://huggingface.co/datasets/tamb2203579/CMU-MOSEI/resolve/main/Raw.zip"
ROOT = "/home/user/data/mosei"
SUBSET = os.path.join(ROOT, "subset.csv")
CLIPS = os.path.join(ROOT, "clips")
STATUS = os.path.join(ROOT, "download_status.csv")
EXCLUDED = os.path.join(ROOT, "download_excluded.csv")
LOG = os.path.join(ROOT, "download.log")
DUR_TOL = 0.5
MAX_ATTEMPTS = 6

_tls = threading.local()
_log_lock = threading.Lock()


def log(msg):
    line = time.strftime("%H:%M:%S") + " " + msg
    with _log_lock:
        print(line, flush=True)
        with open(LOG, "a") as fh:
            fh.write(line + "\n")


def member_name(clip_key):
    return "Raw/" + clip_key + ".mp4"


def dest_path(video_id, clip_id):
    return os.path.join(CLIPS, str(video_id), f"{clip_id}.mp4")


def probe(path):
    """Return (ok, duration, message) for a local mp4 file."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type",
             "-of", "json", path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=120)
        info = json.loads(out.stdout.decode())
    except (subprocess.SubprocessError, ValueError) as e:
        return False, float("nan"), f"ffprobe failed: {e}"
    types = {s.get("codec_type") for s in info.get("streams", [])}
    try:
        dur = float(info.get("format", {}).get("duration", "nan"))
    except ValueError:
        dur = float("nan")
    if "audio" not in types or "video" not in types:
        return False, dur, f"streams {sorted(t for t in types if t)}"
    return True, dur, ""


def get_zip():
    z = getattr(_tls, "zip", None)
    if z is None:
        z = remotezip.RemoteZip(URL)
        _tls.zip = z
    return z


def reset_zip():
    z = getattr(_tls, "zip", None)
    if z is not None:
        try:
            z.close()
        except Exception:
            pass
    _tls.zip = None


def fetch(row, dst):
    """Stream one member to dst (via a .part file). Raises on any transport error."""
    name = member_name(row.clip_key)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    z = get_zip()
    with z.open(name) as src, open(tmp, "wb") as out:
        while True:
            chunk = src.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    os.replace(tmp, dst)


def process(row):
    dst = dest_path(row.video_id, row.clip_id)
    if os.path.isfile(dst):
        ok, dur, msg = probe(dst)
        if ok and abs(dur - float(row.dur)) <= DUR_TOL:
            return dict(clip_key=row.clip_key, video_id=row.video_id, clip_id=row.clip_id,
                        status="ok", attempts=0, probe_dur=dur, error="", cached=1)
        os.remove(dst)  # invalid leftover from an earlier run
    last = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            fetch(row, dst)
            ok, dur, msg = probe(dst)
            if ok and abs(dur - float(row.dur)) <= DUR_TOL:
                return dict(clip_key=row.clip_key, video_id=row.video_id, clip_id=row.clip_id,
                            status="ok", attempts=attempt, probe_dur=dur, error="", cached=0)
            last = f"validation: {msg or f'duration {dur:.2f} vs dur {float(row.dur):.2f}'}"
            if os.path.exists(dst):
                os.remove(dst)
        except KeyError:
            last = "member not in archive"
            break  # not transient
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            reset_zip()  # the HTTP session may be broken; start a fresh RemoteZip
        if attempt < MAX_ATTEMPTS:
            wait = min(120.0, 2.0 ** attempt) + random.uniform(0, 1.0)
            log(f"RETRY {row.clip_key} attempt {attempt} failed ({last}); sleeping {wait:.1f}s")
            time.sleep(wait)
    return dict(clip_key=row.clip_key, video_id=row.video_id, clip_id=row.clip_id,
                status="excluded", attempts=MAX_ATTEMPTS if last != "member not in archive" else 1,
                probe_dur=float("nan"), error=last, cached=0)


def write_outputs(results):
    cols = ["clip_key", "video_id", "clip_id", "status", "attempts", "probe_dur", "error", "cached"]
    with open(STATUS + ".tmp", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in results:
            w.writerow({**r, "probe_dur": repr(float(r["probe_dur"]))})
    os.replace(STATUS + ".tmp", STATUS)
    exc = [r for r in results if r["status"] != "ok"]
    with open(EXCLUDED + ".tmp", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clip_key", "video_id", "clip_id", "attempts", "error"])
        for r in exc:
            w.writerow([r["clip_key"], r["video_id"], r["clip_id"], r["attempts"], r["error"]])
    os.replace(EXCLUDED + ".tmp", EXCLUDED)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only the first N subset rows (testing)")
    args = ap.parse_args()
    sub = pd.read_csv(SUBSET, dtype={"video_id": str, "clip_id": str})
    if args.limit:
        sub = sub.head(args.limit)
    rows = list(sub.itertuples(index=False))
    log(f"start: {len(rows)} clips, {args.threads} threads, url={URL}")
    results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        futs = {pool.submit(process, r): r for r in rows}
        for i, f in enumerate(as_completed(futs), 1):
            r = futs[f]
            try:
                res = f.result()
            except Exception:
                res = dict(clip_key=r.clip_key, video_id=r.video_id, clip_id=r.clip_id, status="excluded",
                           attempts=0, probe_dur=float("nan"), error=traceback.format_exc(limit=2)[-300:],
                           cached=0)
            results.append(res)
            if res["status"] == "ok":
                log(f"[{i}/{len(rows)}] ok {res['clip_key']} attempts={res['attempts']} "
                    f"cached={res['cached']} dur={res['probe_dur']:.2f}")
            else:
                log(f"[{i}/{len(rows)}] EXCLUDED {res['clip_key']}: {res['error']}")
            if i % 100 == 0:
                write_outputs(sorted(results, key=lambda x: x["clip_key"]))
    results.sort(key=lambda x: x["clip_key"])
    write_outputs(results)
    n_ok = sum(r["status"] == "ok" for r in results)
    log(f"DONE in {time.time() - t0:.0f}s: ok={n_ok} excluded={len(results) - n_ok} of {len(results)}")


if __name__ == "__main__":
    sys.exit(main())
