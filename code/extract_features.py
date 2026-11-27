#!/usr/bin/env python3
"""Audio and visual feature extraction for the RAVDESS speech corpus (see SPEC_extraction.md)."""
import argparse
import csv
import glob
import hashlib
import multiprocessing as mp
import os
import pickle
import shutil
import subprocess
import time
import traceback
import zipfile

import numpy as np

RAW_DIR = "/home/user/data/raw"
RAVDESS_DIR = "/home/user/data/ravdess"
OUT_DIR = "/home/user/data/features"
YUNET_PATH = "/home/user/data/face_detection_yunet_2023mar.onnx"
W2V_NAME = "facebook/wav2vec2-base"
CLIP_NAME = "openai/clip-vit-base-patch32"

SR = 16000
FRAME_STEP = 6
FACE_IN = (640, 360)
FACE_SCORE = 0.6
FACE_NMS = 0.3
FACE_TOPK = 5000
EXPAND = 1.15
CROP = 224
TOTAL_THREADS = 4

VCONDS = ["clean", "compress", "lowlight", "blur"]
ACONDS = ["clean", "snr20", "snr10", "snr0"]
SNR_DB = {"snr20": 20.0, "snr10": 10.0, "snr0": 0.0}
ACTORS_ALL = list(range(1, 25))
CLIPS_PER_ACTOR = 60
TOTAL_CLIPS = 1440

META_COLS = ["clip_id", "actor", "sex", "emotion", "intensity", "statement", "repetition",
             "n_frames", "duration_s"] + [f"face_rate_{v}" for v in VCONDS]
TIMING_COLS = ["clip_id", "actor", "threads", "audio_decode_s", "video_decode_face_s",
               "a0_prosody_s", "a1_w2v_s", "v0_mnet_s", "v1_clip_s"]

_S = {}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def clip_seed(clip_id):
    return int(hashlib.md5(clip_id.encode()).hexdigest()[:8], 16)


def zip_path(actor):
    return os.path.join(RAW_DIR, f"Video_Speech_Actor_{actor:02d}.zip")


def actor_dir(actor):
    return os.path.join(RAVDESS_DIR, f"Actor_{actor:02d}")


def pkl_path(actor):
    return os.path.join(OUT_DIR, f"actor_{actor:02d}.pkl")


def zip_complete(actor):
    p = zip_path(actor)
    if not os.path.isfile(p):
        return False
    try:
        with zipfile.ZipFile(p) as zf:
            zf.namelist()
        return True
    except zipfile.BadZipFile:
        return False


def extract_actor(actor):
    d = actor_dir(actor)
    os.makedirs(d, exist_ok=True)
    n_new = 0
    with zipfile.ZipFile(zip_path(actor)) as zf:
        for info in zf.infolist():
            base = os.path.basename(info.filename)
            if info.is_dir() or not (base.startswith("01-") and base.endswith(".mp4")):
                continue
            dst = os.path.join(d, base)
            if os.path.isfile(dst) and os.path.getsize(dst) == info.file_size:
                continue
            with zf.open(info) as src, open(dst + ".tmp", "wb") as out:
                shutil.copyfileobj(src, out)
            os.replace(dst + ".tmp", dst)
            n_new += 1
    return n_new


def init_worker(threads):
    os.environ["OMP_NUM_THREADS"] = str(threads)
    import cv2
    import torch
    import torchvision
    from transformers import AutoFeatureExtractor, CLIPImageProcessor, CLIPModel, Wav2Vec2Model

    torch.set_num_threads(threads)
    cv2.setNumThreads(threads)
    _S["threads"] = threads
    _S["cv2"] = cv2
    _S["torch"] = torch
    _S["w2v"] = Wav2Vec2Model.from_pretrained(W2V_NAME).eval()
    _S["w2v_fe"] = AutoFeatureExtractor.from_pretrained(W2V_NAME)
    _S["clip"] = CLIPModel.from_pretrained(CLIP_NAME).eval()
    _S["clip_proc"] = CLIPImageProcessor.from_pretrained(CLIP_NAME)
    weights = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    _S["mnet"] = torchvision.models.mobilenet_v3_small(weights=weights).eval()
    det = cv2.FaceDetectorYN.create(YUNET_PATH, "", FACE_IN, FACE_SCORE, FACE_NMS, FACE_TOPK)
    det.setInputSize(FACE_IN)
    _S["yunet"] = det
    log(f"worker pid={os.getpid()} models loaded, torch threads={threads}")


# ---------------- audio ----------------

def decode_audio(path):
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path, "-vn",
         "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
        stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def add_awgn(y, snr_db, rng):
    p_sig = np.mean(y.astype(np.float64) ** 2)
    p_noise = p_sig / (10.0 ** (snr_db / 10.0))
    noise = rng.standard_normal(y.shape[0]) * np.sqrt(p_noise)
    return (y + noise).astype(np.float32)


def embed_prosody(y):
    import librosa
    feats = []
    mf = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=40)
    d1 = librosa.feature.delta(mf, order=1)
    d2 = librosa.feature.delta(mf, order=2)
    for m in (mf, d1, d2):
        feats += [m.mean(axis=1), m.std(axis=1)]
    for v in (librosa.feature.rms(y=y)[0],
              librosa.feature.zero_crossing_rate(y=y)[0],
              librosa.feature.spectral_centroid(y=y, sr=SR)[0],
              librosa.feature.spectral_bandwidth(y=y, sr=SR)[0],
              librosa.feature.spectral_rolloff(y=y, sr=SR)[0]):
        feats.append(np.array([v.mean(), v.std()]))
    con = librosa.feature.spectral_contrast(y=y, sr=SR)
    feats += [con.mean(axis=1), con.std(axis=1)]
    chroma = librosa.feature.chroma_stft(y=y, sr=SR)
    feats.append(chroma.mean(axis=1))
    f0 = librosa.yin(y=y, fmin=65, fmax=400, sr=SR)
    feats.append(np.array([f0.mean(), f0.std(), np.percentile(f0, 10), np.percentile(f0, 90)]))
    return np.concatenate(feats).astype(np.float32)


def embed_w2v(y):
    torch = _S["torch"]
    inp = _S["w2v_fe"](y, sampling_rate=SR, return_tensors="pt")
    with torch.inference_mode():
        out = _S["w2v"](inp["input_values"], output_hidden_states=True)
    hs = torch.stack([h[0].mean(dim=0) for h in out.hidden_states]).numpy()
    assert hs.shape == (13, 768), hs.shape
    return hs.astype(np.float16)


# ---------------- video ----------------

def decode_video(path):
    cv2 = _S["cv2"]
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    i = 0
    while True:
        if not cap.grab():
            break
        if i % FRAME_STEP == 0:
            ok, f = cap.retrieve()
            if ok:
                frames.append(f)
        i += 1
    cap.release()
    return frames, i, fps


def degrade(f, cond, rng):
    cv2 = _S["cv2"]
    if cond == "clean":
        return f
    H, W = f.shape[:2]
    if cond == "compress":
        small = cv2.resize(f, (320, 180), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 8])
        dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return cv2.resize(dec, (W, H), interpolation=cv2.INTER_LINEAR)
    if cond == "lowlight":
        x = f.astype(np.float32) / 255.0
        g = 255.0 * (x ** 2.2) * 0.35 + rng.normal(0.0, 6.0, size=f.shape).astype(np.float32)
        return np.clip(np.rint(g), 0, 255).astype(np.uint8)
    if cond == "blur":
        return cv2.GaussianBlur(f, (21, 21), 6)
    raise ValueError(cond)


def face_box(frame):
    H, W = frame.shape[:2]
    small = _S["cv2"].resize(frame, FACE_IN, interpolation=_S["cv2"].INTER_AREA)
    _, faces = _S["yunet"].detect(small)
    if faces is None or len(faces) == 0:
        return None
    best = max(faces, key=lambda r: float(r[2]) * float(r[3]))
    sx, sy = W / FACE_IN[0], H / FACE_IN[1]
    x, y = float(best[0]) * sx, float(best[1]) * sy
    w, h = float(best[2]) * sx, float(best[3]) * sy
    cx, cy = x + w / 2.0, y + h / 2.0
    bw, bh = w * EXPAND, h * EXPAND
    x0 = max(0, int(round(cx - bw / 2.0)))
    x1 = min(W, int(round(cx + bw / 2.0)))
    y0 = max(0, int(round(cy - bh / 2.0)))
    y1 = min(H, int(round(cy + bh / 2.0)))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def crop_face(frame, box):
    cv2 = _S["cv2"]
    if box is None:
        H, W = frame.shape[:2]
        s = min(H, W)
        x0, y0 = (W - s) // 2, (H - s) // 2
        box = (x0, y0, x0 + s, y0 + s)
    x0, y0, x1, y1 = box
    c = cv2.resize(frame[y0:y1, x0:x1], (CROP, CROP), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(c, cv2.COLOR_BGR2RGB)


def face_pass(frames, vcond, rng):
    crops = []
    n_det = 0
    last = None
    for f in frames:
        g = degrade(f, vcond, rng)
        box = face_box(g)
        if box is not None:
            n_det += 1
            last = box
        crops.append(crop_face(g, last))
    return np.stack(crops), n_det


def embed_mnet(crops):
    torch = _S["torch"]
    x = torch.from_numpy(crops).permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (x - mean) / std
    m = _S["mnet"]
    with torch.inference_mode():
        f = m.avgpool(m.features(x)).flatten(1)
    return f.numpy().astype(np.float32)


def embed_clip(crops):
    torch = _S["torch"]
    inp = _S["clip_proc"](images=list(crops), return_tensors="pt")
    with torch.inference_mode():
        f = _S["clip"].get_image_features(pixel_values=inp["pixel_values"])
    if not torch.is_tensor(f):
        f = f.pooler_output
    f = f / f.norm(dim=-1, keepdim=True)
    return f.numpy().astype(np.float32)


# ---------------- per clip / per actor ----------------

def process_clip(path, actor, threads):
    cid = os.path.splitext(os.path.basename(path))[0]
    parts = cid.split("-")
    emotion, intensity, statement, repetition = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
    seed = clip_seed(cid)
    timing = {}

    t = time.perf_counter()
    y = decode_audio(path)
    timing["audio_decode_s"] = time.perf_counter() - t

    audio = {"clean": y}
    for i, ac in enumerate(ACONDS):
        if ac != "clean":
            audio[ac] = add_awgn(y, SNR_DB[ac], np.random.default_rng([seed, 1, i]))

    prosody, w2v = {}, {}
    for ac in ACONDS:
        t = time.perf_counter()
        prosody[ac] = embed_prosody(audio[ac])
        dt_a0 = time.perf_counter() - t
        t = time.perf_counter()
        w2v[ac] = embed_w2v(audio[ac])
        dt_a1 = time.perf_counter() - t
        if ac == "clean":
            timing["a0_prosody_s"] = dt_a0
            timing["a1_w2v_s"] = dt_a1

    t = time.perf_counter()
    frames, n_total, fps = decode_video(path)
    t_dec = time.perf_counter() - t
    if not frames:
        raise RuntimeError(f"no frames decoded for {cid}")
    duration = n_total / fps if fps > 0 else float("nan")

    face_rate, mnet, clipv, clipframes = {}, {}, {}, {}
    for j, vc in enumerate(VCONDS):
        rng = np.random.default_rng([seed, 2, j])
        t = time.perf_counter()
        crops, n_det = face_pass(frames, vc, rng)
        dt_face = time.perf_counter() - t
        if vc == "clean":
            timing["video_decode_face_s"] = t_dec + dt_face

        t = time.perf_counter()
        mf = embed_mnet(crops)
        dt_v0 = time.perf_counter() - t
        t = time.perf_counter()
        cf = embed_clip(crops)
        dt_v1 = time.perf_counter() - t
        if vc == "clean":
            timing["v0_mnet_s"] = dt_v0
            timing["v1_clip_s"] = dt_v1

        face_rate[vc] = n_det / len(frames)
        mnet[vc] = np.concatenate([mf.mean(axis=0), mf.std(axis=0)]).astype(np.float32)
        clipv[vc] = np.concatenate([cf.mean(axis=0), cf.std(axis=0)]).astype(np.float32)
        clipframes[vc] = cf

    rec = dict(
        clip_id=cid, actor=actor, sex="male" if actor % 2 == 1 else "female",
        emotion=emotion, intensity=intensity, statement=statement, repetition=repetition,
        n_frames=len(frames), duration_s=duration, face_rate=face_rate,
        prosody=prosody, w2v=w2v, mnet=mnet, clip=clipv, clipframes=clipframes,
        timing=dict(clip_id=cid, actor=actor, threads=threads, **timing),
    )
    return rec


def process_actor(actor):
    threads = _S["threads"]
    out = pkl_path(actor)
    if os.path.exists(out):
        log(f"actor {actor:02d}: output exists, skipping")
        return actor, 0
    t0 = time.perf_counter()
    n_new = extract_actor(actor)
    files = sorted(glob.glob(os.path.join(actor_dir(actor), "01-*.mp4")))
    log(f"actor {actor:02d}: {len(files)} clips available ({n_new} newly extracted)")
    recs = []
    for f in files:
        t = time.perf_counter()
        r = process_clip(f, actor, threads)
        recs.append(r)
        tm = r["timing"]
        log(f"{r['clip_id']} frames={r['n_frames']} face_clean={r['face_rate']['clean']:.3f} "
            f"audio={tm['audio_decode_s']:.2f}s video={tm['video_decode_face_s']:.2f}s "
            f"a0={tm['a0_prosody_s']:.2f}s a1={tm['a1_w2v_s']:.2f}s "
            f"v0={tm['v0_mnet_s']:.2f}s v1={tm['v1_clip_s']:.2f}s total={time.perf_counter() - t:.2f}s")
    tmp = out + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump({"actor": actor, "threads": threads, "clips": recs}, fh, protocol=4)
    os.replace(tmp, out)
    log(f"actor {actor:02d} done: {len(recs)} clips in {time.perf_counter() - t0:.1f}s")
    return actor, len(recs)


def run_actor_safe(actor):
    try:
        return process_actor(actor)
    except Exception:
        log(f"actor {actor:02d} FAILED\n{traceback.format_exc()}")
        return actor, -1


def run_extract(actors, workers):
    os.makedirs(OUT_DIR, exist_ok=True)
    todo = []
    for a in actors:
        if os.path.exists(pkl_path(a)):
            log(f"actor {a:02d}: already processed")
        elif not zip_complete(a):
            log(f"actor {a:02d}: zip missing or incomplete, skipped")
        else:
            todo.append(a)
    threads = max(1, TOTAL_THREADS // workers)
    log(f"processing actors {todo} with {workers} worker(s), {threads} torch thread(s) each")
    if not todo:
        return
    if workers == 1:
        init_worker(threads)
        for a in todo:
            run_actor_safe(a)
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(workers, initializer=init_worker, initargs=(threads,)) as pool:
            for _ in pool.imap_unordered(run_actor_safe, todo):
                pass
    log("extraction finished")


def merge():
    files = sorted(glob.glob(os.path.join(OUT_DIR, "actor_*.pkl")))
    recs, thread_settings = [], set()
    for f in files:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        recs.extend(d["clips"])
        thread_settings.add(d["threads"])
    recs.sort(key=lambda r: r["clip_id"])
    n = len(recs)
    ids = [r["clip_id"] for r in recs]
    assert len(set(ids)) == n, "duplicate clip ids"
    log(f"merging {len(files)} actor files, {n} clips (expected {TOTAL_CLIPS}); "
        f"torch thread settings used: {sorted(thread_settings)}")

    with open(os.path.join(OUT_DIR, "meta.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(META_COLS)
        for r in recs:
            w.writerow([r["clip_id"], r["actor"], r["sex"], r["emotion"], r["intensity"],
                        r["statement"], r["repetition"], r["n_frames"], repr(float(r["duration_s"]))]
                       + [repr(float(r["face_rate"][v])) for v in VCONDS])

    for ac in ACONDS:
        np.save(os.path.join(OUT_DIR, f"prosody_{ac}.npy"),
                np.stack([r["prosody"][ac] for r in recs]).astype(np.float32))
        np.save(os.path.join(OUT_DIR, f"w2v_{ac}.npy"),
                np.stack([r["w2v"][ac] for r in recs]).astype(np.float16))
    for vc in VCONDS:
        np.save(os.path.join(OUT_DIR, f"mnet_{vc}.npy"),
                np.stack([r["mnet"][vc] for r in recs]).astype(np.float32))
        np.save(os.path.join(OUT_DIR, f"clip_{vc}.npy"),
                np.stack([r["clip"][vc] for r in recs]).astype(np.float32))
        with open(os.path.join(OUT_DIR, f"clipframes_{vc}.pkl"), "wb") as fh:
            pickle.dump([r["clipframes"][vc] for r in recs], fh, protocol=4)

    with open(os.path.join(OUT_DIR, "timings.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(TIMING_COLS)
        for r in recs:
            t = r["timing"]
            w.writerow([t[c] if c in ("clip_id", "actor", "threads") else repr(float(t[c]))
                        for c in TIMING_COLS])
    log(f"merge written to {OUT_DIR}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--actors", type=int, nargs="+", help="actor subset (default: 1..24)")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes over actors")
    ap.add_argument("--merge", action="store_true", help="merge per-actor files into final outputs")
    args = ap.parse_args()
    if args.workers < 1 or args.workers > TOTAL_THREADS:
        ap.error("--workers must be between 1 and 4")
    if args.merge:
        merge()
        return
    run_extract(sorted(set(args.actors or ACTORS_ALL)), args.workers)


if __name__ == "__main__":
    main()
