#!/usr/bin/env python3
"""Extension features (see SPEC_extension.md).

RAVDESS part: quality descriptors (QD) for the existing audio/video conditions, new DEMAND noise
conditions (cafe5, street5) with A0/A1 extractors, and the h264 re-encoding condition with
face detection, V0 and V1 extractors.

CREMA-D part: the CREMA-D subset under clean, SNR, DEMAND noise, compress, lowlight and h264
conditions with A0, A1, V0, V1 and QD.

MOSEI part (see SPEC_mosei.md): the selected CMU-MOSEI clips (subset.csv, downloaded by
mosei_download.py) with the CREMA-D audio and video conditions. Frames are capped at 24 (uniform
subsample of the every-6th-frame set) and downscaled to fit 1280x720 before feature extraction.

Decoding, degradations, face crops and extractors are imported from extract_features.py so the
pipeline is identical to the RAVDESS extraction.
"""
import argparse
import csv
import glob
import json
import multiprocessing as mp
import os
import pickle
import subprocess
import sys
import time
import traceback

import cv2
import librosa
import numpy as np
from scipy.io import wavfile
from scipy.signal import get_window

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import extract_features as ef  # noqa: E402

RAVDESS_FEAT = "/home/user/data/features"
CREMA_DIR = "/home/user/data/cremad"
CREMA_FEAT = "/home/user/data/features_cremad"
MOSEI_DIR = "/home/user/data/mosei"
MOSEI_FEAT = "/home/user/data/features_mosei"
MOSEI_BLOCK = 50
MOSEI_MAX_FRAMES = 24
MOSEI_MAX_W, MOSEI_MAX_H = 1280, 720
MOSEI_TIMING_COLS = ["clip_id", "video_id", "threads", "audio_decode_s", "video_decode_face_s",
                     "a0_prosody_s", "a1_w2v_s", "v0_mnet_s", "v1_clip_s"]
WORK_ROOT = "/home/user/data/features_ext_work"
DEMAND_DIR = "/home/user/data/demand"
DEMAND_FILES = {"cafe5": os.path.join(DEMAND_DIR, "PCAFETER", "ch01.wav"),
                "street5": os.path.join(DEMAND_DIR, "STRAFFIC", "ch01.wav")}
NOISE_SNR_DB = 5.0
NEW_AUDIO = ["cafe5", "street5"]
RAV_QD_AUDIO = ["clean", "snr20", "snr10", "snr0", "cafe5", "street5"]
RAV_QD_VIDEO = ["clean", "compress", "lowlight", "blur", "h264"]
CREMA_QD_AUDIO = ["clean", "snr10", "snr0", "cafe5", "street5"]
CREMA_QD_VIDEO = ["clean", "compress", "lowlight", "h264"]
H264_VF = "scale=320:180"
H264_ARGS = ["-c:v", "libx264", "-b:v", "100k", "-maxrate", "100k", "-bufsize", "200k", "-preset", "veryfast"]
AN, AH = 400, 160  # QD audio frame length and hop (25 ms, 10 ms at 16 kHz)
N_MEL = 8
HF_CUT_HZ = 4000.0
CLIP_THR = 0.99
BLOCK = 8
IQD = ["mean_lum", "sd_lum", "dark_frac", "sat_frac", "log_lap_var", "tenengrad",
       "blockiness", "noise_sd", "hsv_sat"]
AQD = (["ae_p5", "ae_p10", "ae_p50", "ae_p90", "ae_p95", "ae_dr_p95_p10",
        "sf_mean", "sf_p10", "sf_p90"]
       + [f"nf_mel{i}" for i in range(N_MEL)]
       + ["hf_ratio", "spec_entropy", "clip_frac"])
VQD = [f"{img}_{d}_{s}" for img in ("full", "crop") for d in IQD for s in ("mean", "sd")]

_NOISE = {}
_MEL = {}


def log(msg):
    ef.log(msg)


# ---------------- DEMAND noise ----------------

def load_noise():
    out = {}
    for k, p in DEMAND_FILES.items():
        sr, d = wavfile.read(p)
        if sr != ef.SR:
            raise ValueError(f"{p}: sample rate {sr}, expected {ef.SR}")
        if d.ndim > 1:
            d = d[:, 0]
        if np.issubdtype(d.dtype, np.integer):
            d = d.astype(np.float32) / float(np.iinfo(d.dtype).max)
        else:
            d = d.astype(np.float32)
        out[k] = d
    return out


def add_demand_noise(y, noise, snr_db, rng):
    L = y.shape[0]
    if noise.shape[0] < L:
        noise = np.tile(noise, int(np.ceil(L / noise.shape[0])) + 1)
    start = int(rng.integers(0, noise.shape[0] - L + 1))
    seg = noise[start:start + L].astype(np.float64)
    p_sig = np.mean(y.astype(np.float64) ** 2)
    p_seg = np.mean(seg ** 2)
    scale = np.sqrt(p_sig / (p_seg * 10.0 ** (snr_db / 10.0)))
    return (y + seg * scale).astype(np.float32)


def make_audio(y, seed, aconds):
    audio = {}
    for ac in aconds:
        if ac == "clean":
            audio[ac] = y
        elif ac in ef.SNR_DB:
            audio[ac] = ef.add_awgn(y, ef.SNR_DB[ac], np.random.default_rng([seed, 1, ef.ACONDS.index(ac)]))
        else:
            k = NEW_AUDIO.index(ac)
            audio[ac] = add_demand_noise(y, _NOISE[ac], NOISE_SNR_DB, np.random.default_rng([seed, 3, k]))
    return audio


# ---------------- audio quality descriptors ----------------

def _audio_consts():
    if not _MEL:
        _MEL["fb"] = librosa.filters.mel(sr=ef.SR, n_fft=AN, n_mels=N_MEL)
        _MEL["win"] = get_window("hann", AN)
        _MEL["freqs"] = np.fft.rfftfreq(AN, 1.0 / ef.SR)
    return _MEL


def audio_qd(y):
    c = _audio_consts()
    x = y.astype(np.float64)
    if x.shape[0] < AN:
        x = np.pad(x, (0, AN - x.shape[0]))
    fr = librosa.util.frame(x, frame_length=AN, hop_length=AH)  # (AN, T)
    T = fr.shape[1]
    e_db = 10.0 * np.log10(np.mean(fr ** 2, axis=0) + 1e-10)
    P = np.abs(np.fft.rfft(fr * c["win"][:, None], axis=0)) ** 2 + 1e-12  # (201, T)
    sf = np.exp(np.mean(np.log(P), axis=0)) / np.mean(P, axis=0)
    L = 10.0 * np.log10(c["fb"] @ P + 1e-12)  # (N_MEL, T)
    k = max(1, int(round(0.1 * T)))
    idx = np.argsort(e_db)[:k]
    nf = L[:, idx].mean(axis=1)
    hf = float((P[c["freqs"] > HF_CUT_HZ].sum(axis=0) / P.sum(axis=0)).mean())
    p = P / P.sum(axis=0)
    ent = float((-(p * np.log(p)).sum(axis=0) / np.log(P.shape[0])).mean())
    clip = float(np.mean(np.abs(y) > CLIP_THR))
    p5, p10, p50, p90, p95 = np.percentile(e_db, [5, 10, 50, 90, 95])
    sf10, sf90 = np.percentile(sf, [10, 90])
    vec = [p5, p10, p50, p90, p95, p95 - p10, sf.mean(), sf10, sf90] + list(nf) + [hf, ent, clip]
    out = np.asarray(vec, dtype=np.float32)
    assert out.shape[0] == len(AQD)
    return out


# ---------------- video quality descriptors ----------------

def img_qd(img, rgb):
    if rgb:
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    else:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gf = g.astype(np.float32)
    H, W = g.shape
    lap = cv2.Laplacian(g, cv2.CV_32F)
    sx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    dh = np.abs(np.diff(gf, axis=1))  # column c: difference between c and c+1
    dv = np.abs(np.diff(gf, axis=0))  # row r: difference between r and r+1
    mh = (np.arange(W - 1) + 1) % BLOCK == 0
    mv = (np.arange(H - 1) + 1) % BLOCK == 0
    blk_h = dh[:, mh].mean() / (dh[:, ~mh].mean() + 1e-6)
    blk_v = dv[mv, :].mean() / (dv[~mv, :].mean() + 1e-6)
    noise = gf - cv2.medianBlur(g, 3).astype(np.float32)
    return np.array([
        gf.mean(), gf.std(), (g < 30).mean(), (g > 245).mean(),
        np.log(lap.var() + 1e-6), (sx ** 2 + sy ** 2).mean(), 0.5 * (blk_h + blk_v),
        noise.std(), hsv[..., 1].mean()], dtype=np.float64)


def video_qd_vector(full_q, crop_q):
    out = []
    for arr in (np.stack(full_q), np.stack(crop_q)):
        for d in range(len(IQD)):
            out += [arr[:, d].mean(), arr[:, d].std()]
    v = np.asarray(out, dtype=np.float32)
    assert v.shape[0] == len(VQD)
    return v


def video_pass(frames, vcond, rng):
    """Degrade (as extract_features.face_pass), detect faces, crop, and compute QD on full frame and crop."""
    crops, full_q, crop_q = [], [], []
    n_det, last = 0, None
    for f in frames:
        g = f if vcond in ("clean", "h264") else ef.degrade(f, vcond, rng)
        box = ef.face_box(g)
        if box is not None:
            n_det += 1
            last = box
        c = ef.crop_face(g, last)
        crops.append(c)
        full_q.append(img_qd(g, rgb=False))
        crop_q.append(img_qd(c, rgb=True))
    return dict(crops=np.stack(crops), n_det=n_det, qd=video_qd_vector(full_q, crop_q))


def h264_frames(path, W, H, n_orig, cid):
    tmp_dir = os.path.join(WORK_ROOT, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp = os.path.join(tmp_dir, f"{os.getpid()}_{cid}.mp4")
    try:
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", path, "-an",
                        "-vf", H264_VF, "-fps_mode", "passthrough"] + H264_ARGS + [tmp], check=True)
        frames, n_total, _ = ef.decode_video(tmp)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    if n_total != n_orig:
        raise RuntimeError(f"{cid}: h264 re-encode has {n_total} frames, original {n_orig}")
    return [cv2.resize(f, (W, H), interpolation=cv2.INTER_LINEAR) for f in frames]


def clip_embed(crops):
    mf = ef.embed_mnet(crops)
    cf = ef.embed_clip(crops)
    return (np.concatenate([mf.mean(axis=0), mf.std(axis=0)]).astype(np.float32),
            np.concatenate([cf.mean(axis=0), cf.std(axis=0)]).astype(np.float32), cf)


# ---------------- per clip ----------------

def ravdess_clip(it):
    cid, path = it["cid"], it["path"]
    seed = ef.clip_seed(cid)
    y = ef.decode_audio(path)
    audio = make_audio(y, seed, RAV_QD_AUDIO)
    qda = {ac: audio_qd(audio[ac]) for ac in RAV_QD_AUDIO}
    prosody, w2v = {}, {}
    for ac in NEW_AUDIO:
        prosody[ac] = ef.embed_prosody(audio[ac])
        w2v[ac] = ef.embed_w2v(audio[ac])

    frames, n_total, fps = ef.decode_video(path)
    if not frames:
        raise RuntimeError(f"no frames decoded for {cid}")
    H, W = frames[0].shape[:2]
    qdv, face_rate = {}, {}
    for j, vc in enumerate(ef.VCONDS):
        res = video_pass(frames, vc, np.random.default_rng([seed, 2, j]))
        qdv[vc] = res["qd"]
        face_rate[vc] = res["n_det"] / len(frames)
    hf = h264_frames(path, W, H, n_total, cid)
    res = video_pass(hf, "h264", None)
    qdv["h264"] = res["qd"]
    face_rate["h264"] = res["n_det"] / len(hf)
    mnet, clipv, cf = clip_embed(res["crops"])
    return dict(clip_id=cid, actor=it["actor"], n_frames=len(frames), qda=qda, qdv=qdv,
                prosody=prosody, w2v=w2v, face_rate=face_rate, mnet=mnet, clip=clipv, clipframes=cf)


def cremad_clip(it):
    cid, path = it["cid"], it["path"]
    seed = ef.clip_seed(cid)
    timing = {}
    t = time.perf_counter()
    y = ef.decode_audio(path)
    timing["audio_decode_s"] = time.perf_counter() - t
    audio = make_audio(y, seed, CREMA_QD_AUDIO)
    qda, prosody, w2v = {}, {}, {}
    for ac in CREMA_QD_AUDIO:
        qda[ac] = audio_qd(audio[ac])
        t = time.perf_counter()
        prosody[ac] = ef.embed_prosody(audio[ac])
        dt_a0 = time.perf_counter() - t
        t = time.perf_counter()
        w2v[ac] = ef.embed_w2v(audio[ac])
        dt_a1 = time.perf_counter() - t
        if ac == "clean":
            timing["a0_prosody_s"] = dt_a0
            timing["a1_w2v_s"] = dt_a1

    t = time.perf_counter()
    frames, n_total, fps = ef.decode_video(path)
    t_dec = time.perf_counter() - t
    if not frames:
        raise RuntimeError(f"no frames decoded for {cid}")
    H, W = frames[0].shape[:2]
    duration = n_total / fps if fps > 0 else float("nan")

    face_rate, qdv, mnet, clipv, clipframes = {}, {}, {}, {}, {}
    for vc in CREMA_QD_VIDEO:
        if vc == "h264":
            src = h264_frames(path, W, H, n_total, cid)
            rng = None
        else:
            src = frames
            rng = np.random.default_rng([seed, 2, ef.VCONDS.index(vc)])
        t = time.perf_counter()
        res = video_pass(src, vc, rng)
        dt_face = time.perf_counter() - t
        if vc == "clean":
            timing["video_decode_face_s"] = t_dec + dt_face
        t = time.perf_counter()
        mf = ef.embed_mnet(res["crops"])
        dt_v0 = time.perf_counter() - t
        t = time.perf_counter()
        cf = ef.embed_clip(res["crops"])
        dt_v1 = time.perf_counter() - t
        if vc == "clean":
            timing["v0_mnet_s"] = dt_v0
            timing["v1_clip_s"] = dt_v1
        face_rate[vc] = res["n_det"] / len(src)
        qdv[vc] = res["qd"]
        mnet[vc] = np.concatenate([mf.mean(axis=0), mf.std(axis=0)]).astype(np.float32)
        clipv[vc] = np.concatenate([cf.mean(axis=0), cf.std(axis=0)]).astype(np.float32)
        clipframes[vc] = cf
    return dict(clip_id=cid, actor=it["actor"], sex=it["sex"], sentence=it["sentence"],
                emotion=it["emotion"], level=it["level"], n_frames=len(frames), duration_s=duration,
                qda=qda, qdv=qdv, prosody=prosody, w2v=w2v, face_rate=face_rate,
                mnet=mnet, clip=clipv, clipframes=clipframes,
                timing=dict(clip_id=cid, actor=it["actor"], threads=_threads(), **timing))


def _threads():
    return ef._S["threads"]


# ---------------- CMU-MOSEI ----------------

def mosei_cid(clip_key):
    return clip_key.replace("/", "_")


def cap_frames(frames):
    """Uniform subsample to at most MOSEI_MAX_FRAMES frames (deterministic in the list length)."""
    n = len(frames)
    if n <= MOSEI_MAX_FRAMES:
        return list(frames)
    idx = np.linspace(0, n - 1, MOSEI_MAX_FRAMES).round().astype(int)
    return [frames[i] for i in idx]


def fit_frames(frames):
    """Downscale all frames to fit within 1280x720 (aspect kept); unchanged if already within."""
    H, W = frames[0].shape[:2]
    if W <= MOSEI_MAX_W and H <= MOSEI_MAX_H:
        return frames
    s = min(MOSEI_MAX_W / W, MOSEI_MAX_H / H)
    nW, nH = max(1, int(round(W * s))), max(1, int(round(H * s)))
    cv2 = ef._S["cv2"]
    return [cv2.resize(f, (nW, nH), interpolation=cv2.INTER_AREA) for f in frames]


def mosei_clip(it):
    cid, path = it["cid"], it["path"]
    seed = ef.clip_seed(cid)
    timing = {}
    t = time.perf_counter()
    y = ef.decode_audio(path)
    timing["audio_decode_s"] = time.perf_counter() - t
    audio = make_audio(y, seed, CREMA_QD_AUDIO)
    qda, prosody, w2v = {}, {}, {}
    for ac in CREMA_QD_AUDIO:
        qda[ac] = audio_qd(audio[ac])
        t = time.perf_counter()
        prosody[ac] = ef.embed_prosody(audio[ac])
        dt_a0 = time.perf_counter() - t
        t = time.perf_counter()
        w2v[ac] = ef.embed_w2v(audio[ac])
        dt_a1 = time.perf_counter() - t
        if ac == "clean":
            timing["a0_prosody_s"] = dt_a0
            timing["a1_w2v_s"] = dt_a1

    t = time.perf_counter()
    frames, n_total, fps = ef.decode_video(path)
    t_dec = time.perf_counter() - t
    if not frames:
        raise RuntimeError(f"no frames decoded for {cid}")
    H0, W0 = frames[0].shape[:2]
    frames = fit_frames(cap_frames(frames))
    H, W = frames[0].shape[:2]
    duration = n_total / fps if fps > 0 else float("nan")

    face_rate, qdv, mnet, clipv, clipframes = {}, {}, {}, {}, {}
    for vc in CREMA_QD_VIDEO:
        if vc == "h264":
            # h264_frames returns every-6th frames at (W, H); apply the same cap as the clean set
            src = cap_frames(h264_frames(path, W, H, n_total, cid))
            rng = None
        else:
            src = frames
            rng = np.random.default_rng([seed, 2, ef.VCONDS.index(vc)])
        t = time.perf_counter()
        res = video_pass(src, vc, rng)
        dt_face = time.perf_counter() - t
        if vc == "clean":
            timing["video_decode_face_s"] = t_dec + dt_face
        t = time.perf_counter()
        mf = ef.embed_mnet(res["crops"])
        dt_v0 = time.perf_counter() - t
        t = time.perf_counter()
        cf = ef.embed_clip(res["crops"])
        dt_v1 = time.perf_counter() - t
        if vc == "clean":
            timing["v0_mnet_s"] = dt_v0
            timing["v1_clip_s"] = dt_v1
        face_rate[vc] = res["n_det"] / len(src)
        qdv[vc] = res["qd"]
        mnet[vc] = np.concatenate([mf.mean(axis=0), mf.std(axis=0)]).astype(np.float32)
        clipv[vc] = np.concatenate([cf.mean(axis=0), cf.std(axis=0)]).astype(np.float32)
        clipframes[vc] = cf
    return dict(clip_id=cid, video_id=it["video_id"], clip_num=it["clip_num"], split=it["split"],
                emotion=it["emotion"], cls=it["cls"], dur=it["dur"], n_frames=len(frames),
                duration_s=duration, width=W0, height=H0, proc_width=W, proc_height=H,
                qda=qda, qdv=qdv, prosody=prosody, w2v=w2v, face_rate=face_rate,
                mnet=mnet, clip=clipv, clipframes=clipframes,
                timing=dict(clip_id=cid, video_id=it["video_id"], threads=_threads(), **timing))


# ---------------- groups (RAVDESS actor, CREMA-D block, MOSEI block) ----------------

def run_group(job):
    out, kind, items = job["out"], job["kind"], job["items"]
    if os.path.exists(out):
        log(f"{os.path.basename(out)}: exists, skipping")
        return out, 0, 0
    t0 = time.perf_counter()
    recs, fails = [], []
    for it in items:
        t = time.perf_counter()
        try:
            if kind == "ravdess":
                r = ravdess_clip(it)
            elif kind == "mosei":
                r = mosei_clip(it)
            else:
                r = cremad_clip(it)
        except Exception as e:
            if kind == "ravdess":
                raise
            fails.append((it["cid"], f"{type(e).__name__}: {e}"))
            log(f"EXCLUDED {it['cid']}: {e}")
            continue
        recs.append(r)
        log(f"{it['cid']} frames={r['n_frames']} clip_time={time.perf_counter() - t:.2f}s")
    tmp = out + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump({"threads": ef._S["threads"], "clips": recs, "failed": fails}, fh, protocol=4)
    os.replace(tmp, out)
    log(f"{os.path.basename(out)} done: {len(recs)} clips, {len(fails)} failed, "
        f"{time.perf_counter() - t0:.1f}s")
    return out, len(recs), len(fails)


def run_job_safe(job):
    try:
        return run_group(job)
    except Exception:
        log(f"{os.path.basename(job['out'])} FAILED\n{traceback.format_exc()}")
        return job["out"], -1, -1


def init_worker(threads):
    ef.init_worker(threads)
    _NOISE.update(load_noise())
    _audio_consts()


def read_ravdess_meta():
    with open(os.path.join(RAVDESS_FEAT, "meta.csv"), newline="") as fh:
        return [(r["clip_id"], int(r["actor"])) for r in csv.DictReader(fh)]


def crema_subset():
    with open(os.path.join(CREMA_DIR, "subset_files.txt")) as fh:
        names = [ln.strip() for ln in fh if ln.strip()]
    return sorted(set(names))


def crema_sex():
    out = {}
    with open(os.path.join(CREMA_DIR, "meta", "VideoDemographics.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            out[int(r["ActorID"])] = r["Sex"].strip().lower()
    return out


CREMA_EMO = {"NEU": 1, "HAP": 3, "SAD": 4, "ANG": 5, "FEA": 6, "DIS": 7}


def mosei_items(limit):
    """Downloaded (status ok) subset clips in subset.csv order."""
    with open(os.path.join(MOSEI_DIR, "download_status.csv"), newline="") as fh:
        status = {r["clip_key"]: r["status"] for r in csv.DictReader(fh)}
    items = []
    with open(os.path.join(MOSEI_DIR, "subset.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            if status.get(r["clip_key"]) != "ok":
                continue
            items.append(dict(cid=mosei_cid(r["clip_key"]), path=os.path.join(
                MOSEI_DIR, "clips", r["video_id"], r["clip_id"] + ".mp4"), video_id=r["video_id"],
                clip_num=int(r["clip_id"]), split=r["split"], emotion=int(r["emotion"]),
                cls=r["cls"], dur=float(r["dur"])))
    if limit:
        items = items[:limit]
    return items


def build_jobs(part, workers, limit, work):
    jobs = []
    if part == "ravdess":
        rows = read_ravdess_meta()
        if limit:
            rows = rows[:limit]
        by_actor = {}
        for cid, actor in rows:
            it = dict(cid=cid, actor=actor, path=os.path.join(ef.actor_dir(actor), cid + ".mp4"))
            by_actor.setdefault(actor, []).append(it)
        for actor in sorted(by_actor):
            jobs.append(dict(out=os.path.join(work, f"actor_{actor:02d}.pkl"), kind="ravdess",
                             items=by_actor[actor]))
    elif part == "mosei":
        items = mosei_items(limit)
        for b in range(0, len(items), MOSEI_BLOCK):
            jobs.append(dict(out=os.path.join(work, f"block_{b // MOSEI_BLOCK:03d}.pkl"), kind="mosei",
                             items=items[b:b + MOSEI_BLOCK]))
    else:
        names = crema_subset()
        if limit:
            names = names[:limit]
        sex = crema_sex()
        items = []
        for cid in names:
            p = cid.split("_")
            path = os.path.join(CREMA_DIR, "flv", cid + ".flv")
            if not os.path.isfile(path):
                log(f"missing video {cid}, excluded")
                continue
            items.append(dict(cid=cid, actor=int(p[0]), path=path, sex=sex[int(p[0])],
                              sentence=p[1], emotion=CREMA_EMO[p[2]], level=p[3]))
        block = 200
        for b in range(0, len(items), block):
            jobs.append(dict(out=os.path.join(work, f"block_{b // block:03d}.pkl"), kind="cremad",
                             items=items[b:b + block]))
    return jobs


# ---------------- merge ----------------

def _safe_write(path):
    if os.path.exists(path):
        raise FileExistsError(f"refusing to overwrite existing file {path}")
    return path


def _load_pkls(work):
    recs, fails, threads = [], [], set()
    for f in sorted(glob.glob(os.path.join(work, "*.pkl"))):
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        recs.extend(d["clips"])
        fails.extend(d.get("failed", []))
        threads.add(d["threads"])
    recs.sort(key=lambda r: r["clip_id"])
    return recs, fails, threads


def _save_qd_names(out_dir):
    p = _safe_write(os.path.join(out_dir, "qd_names.json"))
    with open(p, "w") as fh:
        json.dump({"audio": AQD, "video": VQD}, fh, indent=1)


def merge_ravdess(work, out_dir):
    recs, fails, threads = _load_pkls(work)
    meta_ids = [cid for cid, _ in read_ravdess_meta()]
    got = [r["clip_id"] for r in recs]
    if got != meta_ids:
        raise RuntimeError(f"clip set/order mismatch: got {len(got)}, expected {len(meta_ids)}")
    log(f"RAVDESS merge: {len(recs)} clips, thread settings {sorted(threads)}")
    _save_qd_names(out_dir)
    with open(_safe_write(os.path.join(out_dir, "meta_ext.csv")), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clip_id", "face_rate_h264"])
        for r in recs:
            w.writerow([r["clip_id"], repr(float(r["face_rate"]["h264"]))])
    for ac in RAV_QD_AUDIO:
        np.save(_safe_write(os.path.join(out_dir, f"qda_{ac}.npy")),
                np.stack([r["qda"][ac] for r in recs]).astype(np.float32))
    for vc in RAV_QD_VIDEO:
        np.save(_safe_write(os.path.join(out_dir, f"qdv_{vc}.npy")),
                np.stack([r["qdv"][vc] for r in recs]).astype(np.float32))
    for ac in NEW_AUDIO:
        np.save(_safe_write(os.path.join(out_dir, f"prosody_{ac}.npy")),
                np.stack([r["prosody"][ac] for r in recs]).astype(np.float32))
        np.save(_safe_write(os.path.join(out_dir, f"w2v_{ac}.npy")),
                np.stack([r["w2v"][ac] for r in recs]).astype(np.float16))
    np.save(_safe_write(os.path.join(out_dir, "mnet_h264.npy")),
            np.stack([r["mnet"] for r in recs]).astype(np.float32))
    np.save(_safe_write(os.path.join(out_dir, "clip_h264.npy")),
            np.stack([r["clip"] for r in recs]).astype(np.float32))
    with open(_safe_write(os.path.join(out_dir, "clipframes_h264.pkl")), "wb") as fh:
        pickle.dump([r["clipframes"] for r in recs], fh, protocol=4)
    log(f"RAVDESS merge written to {out_dir}")


def merge_cremad(work, out_dir):
    recs, fails, threads = _load_pkls(work)
    if not recs:
        raise RuntimeError(f"no processed CREMA-D clips in {work}")
    subset = crema_subset()
    done = {r["clip_id"] for r in recs}
    failed = dict(fails)
    excluded = []
    for cid in subset:
        if cid in done:
            continue
        if cid in failed:
            excluded.append((cid, failed[cid]))
        elif not os.path.isfile(os.path.join(CREMA_DIR, "flv", cid + ".flv")):
            excluded.append((cid, "missing flv file"))
        else:
            excluded.append((cid, "not processed"))
    log(f"CREMA-D merge: {len(recs)} clips, {len(excluded)} excluded, thread settings {sorted(threads)}")
    _save_qd_names(out_dir)
    with open(_safe_write(os.path.join(out_dir, "excluded.csv")), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clip_id", "reason"])
        for cid, why in excluded:
            w.writerow([cid, why])
    with open(_safe_write(os.path.join(out_dir, "meta.csv")), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clip_id", "actor", "sex", "emotion", "sentence", "level", "n_frames", "duration_s"]
                   + [f"face_rate_{v}" for v in CREMA_QD_VIDEO])
        for r in recs:
            w.writerow([r["clip_id"], r["actor"], r["sex"], r["emotion"], r["sentence"], r["level"],
                        r["n_frames"], repr(float(r["duration_s"]))]
                       + [repr(float(r["face_rate"][v])) for v in CREMA_QD_VIDEO])
    for ac in CREMA_QD_AUDIO:
        np.save(_safe_write(os.path.join(out_dir, f"qda_{ac}.npy")),
                np.stack([r["qda"][ac] for r in recs]).astype(np.float32))
        np.save(_safe_write(os.path.join(out_dir, f"prosody_{ac}.npy")),
                np.stack([r["prosody"][ac] for r in recs]).astype(np.float32))
        np.save(_safe_write(os.path.join(out_dir, f"w2v_{ac}.npy")),
                np.stack([r["w2v"][ac] for r in recs]).astype(np.float16))
    for vc in CREMA_QD_VIDEO:
        np.save(_safe_write(os.path.join(out_dir, f"qdv_{vc}.npy")),
                np.stack([r["qdv"][vc] for r in recs]).astype(np.float32))
        np.save(_safe_write(os.path.join(out_dir, f"mnet_{vc}.npy")),
                np.stack([r["mnet"][vc] for r in recs]).astype(np.float32))
        np.save(_safe_write(os.path.join(out_dir, f"clip_{vc}.npy")),
                np.stack([r["clip"][vc] for r in recs]).astype(np.float32))
        with open(_safe_write(os.path.join(out_dir, f"clipframes_{vc}.pkl")), "wb") as fh:
            pickle.dump([r["clipframes"][vc] for r in recs], fh, protocol=4)
    with open(_safe_write(os.path.join(out_dir, "timings.csv")), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(ef.TIMING_COLS)
        for r in recs:
            t = r["timing"]
            w.writerow([t[c] if c in ("clip_id", "actor", "threads") else repr(float(t[c]))
                        for c in ef.TIMING_COLS])
    log(f"CREMA-D merge written to {out_dir}")


def merge_mosei(work, out_dir):
    recs, fails, threads = _load_pkls(work)
    if not recs:
        raise RuntimeError(f"no processed MOSEI clips in {work}")
    with open(os.path.join(MOSEI_DIR, "download_status.csv"), newline="") as fh:
        dl = {r["clip_key"]: r for r in csv.DictReader(fh)}
    done = {r["clip_id"] for r in recs}
    failed = dict(fails)
    excluded = []
    with open(os.path.join(MOSEI_DIR, "subset.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            cid = mosei_cid(r["clip_key"])
            if cid in done:
                continue
            st = dl.get(r["clip_key"])
            if st is None:
                why = "not in download status"
            elif st["status"] != "ok":
                why = "download: " + st["error"]
            elif cid in failed:
                why = failed[cid]
            else:
                why = "not processed"
            excluded.append((cid, r["clip_key"], why))
    log(f"MOSEI merge: {len(recs)} clips, {len(excluded)} excluded, thread settings {sorted(threads)}")
    _save_qd_names(out_dir)
    with open(_safe_write(os.path.join(out_dir, "excluded.csv")), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clip_id", "clip_key", "reason"])
        for cid, key, why in excluded:
            w.writerow([cid, key, why])
    with open(_safe_write(os.path.join(out_dir, "meta.csv")), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clip_id", "video_id", "clip_num", "split", "emotion", "cls", "dur", "duration_s",
                    "n_frames", "width", "height", "proc_width", "proc_height"]
                   + [f"face_rate_{v}" for v in CREMA_QD_VIDEO])
        for r in recs:
            w.writerow([r["clip_id"], r["video_id"], r["clip_num"], r["split"], r["emotion"], r["cls"],
                        repr(float(r["dur"])), repr(float(r["duration_s"])), r["n_frames"], r["width"],
                        r["height"], r["proc_width"], r["proc_height"]]
                       + [repr(float(r["face_rate"][v])) for v in CREMA_QD_VIDEO])
    for ac in CREMA_QD_AUDIO:
        np.save(_safe_write(os.path.join(out_dir, f"qda_{ac}.npy")),
                np.stack([r["qda"][ac] for r in recs]).astype(np.float32))
        np.save(_safe_write(os.path.join(out_dir, f"prosody_{ac}.npy")),
                np.stack([r["prosody"][ac] for r in recs]).astype(np.float32))
        np.save(_safe_write(os.path.join(out_dir, f"w2v_{ac}.npy")),
                np.stack([r["w2v"][ac] for r in recs]).astype(np.float16))
    for vc in CREMA_QD_VIDEO:
        np.save(_safe_write(os.path.join(out_dir, f"qdv_{vc}.npy")),
                np.stack([r["qdv"][vc] for r in recs]).astype(np.float32))
        np.save(_safe_write(os.path.join(out_dir, f"mnet_{vc}.npy")),
                np.stack([r["mnet"][vc] for r in recs]).astype(np.float32))
        np.save(_safe_write(os.path.join(out_dir, f"clip_{vc}.npy")),
                np.stack([r["clip"][vc] for r in recs]).astype(np.float32))
        with open(_safe_write(os.path.join(out_dir, f"clipframes_{vc}.pkl")), "wb") as fh:
            pickle.dump([r["clipframes"][vc] for r in recs], fh, protocol=4)
    with open(_safe_write(os.path.join(out_dir, "timings.csv")), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(MOSEI_TIMING_COLS)
        for r in recs:
            t = r["timing"]
            w.writerow([t[c] if c in ("clip_id", "video_id", "threads") else repr(float(t[c]))
                        for c in MOSEI_TIMING_COLS])
    log(f"MOSEI merge written to {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["ravdess", "cremad", "mosei"], required=True)
    ap.add_argument("--workers", type=int, default=2, help="parallel processes (threads = 4 // workers)")
    ap.add_argument("--merge", action="store_true", help="merge per-group files into final outputs")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N clips (testing)")
    ap.add_argument("--work", default=None, help="per-group work directory")
    ap.add_argument("--out", default=None, help="output directory for --merge")
    args = ap.parse_args()
    if args.workers < 1 or args.workers > ef.TOTAL_THREADS:
        ap.error(f"--workers must be between 1 and {ef.TOTAL_THREADS}")
    work = args.work or os.path.join(WORK_ROOT, args.part)
    out_dir = args.out or {"ravdess": RAVDESS_FEAT, "cremad": CREMA_FEAT, "mosei": MOSEI_FEAT}[args.part]
    os.makedirs(work, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    if args.merge:
        {"ravdess": merge_ravdess, "cremad": merge_cremad, "mosei": merge_mosei}[args.part](work, out_dir)
        return
    threads = max(1, ef.TOTAL_THREADS // args.workers)
    jobs = build_jobs(args.part, args.workers, args.limit, work)
    todo = [j for j in jobs if not os.path.exists(j["out"])]
    log(f"{args.part}: {len(jobs)} groups, {len(todo)} to run, {args.workers} worker(s), "
        f"{threads} torch thread(s) each")
    if not todo:
        return
    if args.workers == 1:
        init_worker(threads)
        for j in todo:
            run_job_safe(j)
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(args.workers, initializer=init_worker, initargs=(threads,)) as pool:
            for out, n, nf in pool.imap_unordered(run_job_safe, todo):
                log(f"group finished: {os.path.basename(out)} clips={n} failed={nf}")
    log("extraction finished")


if __name__ == "__main__":
    main()
