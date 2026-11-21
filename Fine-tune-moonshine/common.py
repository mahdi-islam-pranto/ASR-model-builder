"""
Shared code for the fine-tuning scripts: setup/logging, text cleaning, audio decoding, phone simulation,
model loading, the training-batch collator, and evaluation (transcribe + WER/CER).
"""

import argparse
import io
import os
import random
import re
import sys
import tempfile
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np

import config as C

SR = 16000
START_ID, EOS_ID, PAD_ID = 2, 3, 0      # [CLS], [SEP], [PAD]. Checked against the model; see README


# =============================================================== setup / logging
class _Tee:
    """Write everything printed to the console into a log file too."""

    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, s):
        self.stream.write(s)
        self.fh.write(s)
        self.fh.flush()

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self.stream, name)


def load_dotenv():
    """Read HF_TOKEN etc. from the project's .env file (if the variable is not already set)."""
    for env_file in (C.HERE / ".env", C.PROJECT_ROOT / ".env"):
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def base_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--quick", action="store_true", help="tiny end-to-end test run (do this first)")
    p.add_argument("--run-name", help="override RUN_NAME from config.py")
    return p


def setup(args, step):
    """Apply --quick / --run-name, create folders, log to runs/<run>/logs/<step>_<time>.log, seed."""
    if getattr(args, "quick", False) and not C.QUICK:
        C.apply_quick()
    if getattr(args, "run_name", None):
        C.RUN_NAME = args.run_name
    for d in (C.raw_dir(), C.processed_dir(), C.run_dir() / "logs"):
        d.mkdir(parents=True, exist_ok=True)
    if not isinstance(sys.stdout, _Tee):
        log = C.run_dir() / "logs" / f"{step}_{datetime.now():%Y%m%d_%H%M%S}.log"
        fh = open(log, "a", encoding="utf-8")
        sys.stdout, sys.stderr = _Tee(sys.stdout, fh), _Tee(sys.stderr, fh)
        print(f"log file: {log}")
    for s in (sys.stdout, sys.stderr):
        try:
            s.stream.reconfigure(encoding="utf-8")   # Bangla text on a Windows console
        except Exception:
            pass
    load_dotenv()
    random.seed(C.SEED)
    np.random.seed(C.SEED)
    import torch
    torch.manual_seed(C.SEED)
    print(f"=== {step} | run={C.RUN_NAME} | quick={C.QUICK} | {datetime.now():%Y-%m-%d %H:%M:%S}")


def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def precision():
    """'bf16' | 'fp16' | 'fp32'. bf16 (Ampere+ GPUs: RTX 30xx/40xx, A100, ...) is more stable than fp16."""
    import torch
    if device() != "cuda":
        return "fp32"
    if C.PRECISION != "auto":
        return C.PRECISION
    return "bf16" if torch.cuda.is_bf16_supported() else "fp16"


# =============================================================== text
_TAGS = re.compile(r"\[[^\]]*\]|<[^>]*>|\{[^}]*\}")
_PUNCT = re.compile(r"[।॥,\.\?!;:\"'‘’“”\-–—\(\)/\\…*_~`|=+#@&%^]")
_ZW = re.compile(r"[​‌‍﻿]")
_LATIN = re.compile(r"[A-Za-z]")
_DIGIT = re.compile(r"[0-9০-৯]")


def normalize_text(s):
    """One cleaning rule for training labels AND scoring: NFC, no tags, no punctuation, single spaces."""
    s = unicodedata.normalize("NFC", s or "")
    s = _TAGS.sub(" ", s)
    s = _ZW.sub("", s)
    s = _PUNCT.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def check_clip(tokenizer, text, dur):
    """Returns (clean text, label ids) or (None, reason why the clip is dropped)."""
    t = normalize_text(text)
    if not t:
        return None, "empty text"
    if dur < C.MIN_SEC:
        return None, "too short"
    if dur > C.MAX_SEC:
        return None, "too long"
    if C.DROP_LATIN and _LATIN.search(t):
        return None, "latin letters"
    if C.DROP_DIGITS and _DIGIT.search(t):
        return None, "digits"
    wps = len(t.split()) / dur
    if dur > 3 and (wps > 6 or wps < 0.3):
        return None, "words/sec implausible"
    ids = tokenizer(t, add_special_tokens=False)["input_ids"] + [EOS_ID]   # labels: tokens + [SEP], no [CLS]
    if len(ids) > C.MAX_LABEL_TOKENS:
        return None, "too many tokens"
    if tokenizer.unk_token_id in ids:
        return None, "unknown token"
    return t, ids


# =============================================================== audio
def decode_audio(b):
    """Audio file bytes -> (16 kHz mono float32, native sample rate)."""
    import librosa
    import soundfile as sf
    try:
        y, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
        y = y.mean(axis=1)
    except Exception:
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(b)
            tmp = f.name
        try:
            y, sr = librosa.load(tmp, sr=None, mono=True)
        finally:
            os.remove(tmp)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y.astype(np.float32), int(sr)


def to_int16(y):
    return np.clip(y * 32767, -32768, 32767).astype(np.int16)


_SOS_PHONE = None


def phone_sim(y, rng):
    """16 kHz clip -> phone-line version (returned at 16 kHz, like our real calls after upsampling)."""
    global _SOS_PHONE
    from scipy.signal import butter, resample_poly, sosfilt
    if _SOS_PHONE is None:
        _SOS_PHONE = butter(4, [300, 3400], btype="bandpass", fs=8000, output="sos")
    y8 = resample_poly(y, 1, 2)                                   # 16 -> 8 kHz: nothing above 4 kHz survives
    y8 = sosfilt(_SOS_PHONE, y8)                                  # phone band 300-3400 Hz
    y8 = y8 / (np.abs(y8).max() + 1e-9) * rng.uniform(0.3, 0.9)   # random level
    mu = 255.0                                                    # G.711 mu-law codec (8 bit)
    c = np.sign(y8) * np.log1p(mu * np.abs(y8)) / np.log1p(mu)
    c = np.round((c + 1) / 2 * mu) / mu * 2 - 1
    y8 = np.sign(c) * ((1 + mu) ** np.abs(c) - 1) / mu
    snr_db = rng.uniform(15, 35)                                  # line / room noise
    p = np.mean(y8 ** 2) + 1e-12
    y8 = y8 + rng.normal(0, np.sqrt(p / 10 ** (snr_db / 10)), len(y8))
    return np.clip(resample_poly(y8, 2, 1), -1, 1).astype(np.float32)


# =============================================================== downloads / parquet
def download(repo, files):
    """Download dataset files to data/raw/<repo> (skips files that are already there)."""
    from huggingface_hub import hf_hub_download
    out = []
    for f in files:
        try:
            p = hf_hub_download(repo_id=repo, filename=f, repo_type="dataset",
                                local_dir=str(C.raw_dir() / repo.replace("/", "__")))
        except Exception as e:
            msg = str(e)
            if "gated" in msg.lower() or "401" in msg or "403" in msg:
                raise SystemExit(
                    f"\nNo access to {repo}.\n  1. Open https://huggingface.co/datasets/{repo} and accept the conditions"
                    f"\n  2. Put HF_TOKEN=<your read token> in the project .env (or `huggingface-cli login`)\n") from e
            raise
        out.append(p)
        print(f"  ok {f} ({os.path.getsize(p) / 1e6:.0f} MB)")
    return out


def find_audio_col(schema):
    import pyarrow as pa
    for f in schema:
        if pa.types.is_struct(f.type) and "bytes" in [c.name for c in f.type]:
            return f.name
    raise ValueError(f"no audio column (struct with 'bytes') in {schema.names}")


def string_cols(schema):
    import pyarrow as pa
    return [f.name for f in schema if pa.types.is_string(f.type) or pa.types.is_large_string(f.type)]


def iter_clips(paths, audio_col, text_col, keep=None, category_col=None):
    """Yields (16 kHz float audio, native sr, raw text, category) from parquet files, one row group at a time."""
    import pyarrow.parquet as pq
    cols = list(dict.fromkeys([audio_col, text_col] + ([category_col] if category_col else []) + list(keep or {})))
    for p in paths:
        pf = pq.ParquetFile(p)
        for g in range(pf.num_row_groups):
            for r in pf.read_row_group(g, columns=cols).to_pylist():
                if keep and any(r.get(k) not in v for k, v in keep.items()):
                    continue
                b = (r[audio_col] or {}).get("bytes")
                if not b:
                    continue
                y, sr = decode_audio(b)
                yield y, sr, r[text_col], (str(r.get(category_col)) if category_col else "")


# =============================================================== model
def load_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(C.BASE_MODEL))
    assert tok.convert_ids_to_tokens([PAD_ID, START_ID, EOS_ID]) == ["[PAD]", "[CLS]", "[SEP]"]
    return tok


def fixed_generation_config():
    """The model's saved generation_config.json has wrong ids (1/2/2). These are the real ones."""
    from transformers import GenerationConfig
    return GenerationConfig(decoder_start_token_id=START_ID, bos_token_id=START_ID,
                            eos_token_id=EOS_ID, pad_token_id=PAD_ID)


def load_model(path=None):
    import torch
    from transformers import AutoModelForSpeechSeq2Seq
    m = AutoModelForSpeechSeq2Seq.from_pretrained(str(path or C.BASE_MODEL), dtype=torch.float32)
    m.generation_config = fixed_generation_config()
    return m.to(device())


# =============================================================== datasets on disk
def features():
    from datasets import Features, Value
    try:
        from datasets import List as _List
    except ImportError:
        from datasets import Sequence as _List
    return Features({
        "audio": _List(Value("int16")), "text": Value("string"), "labels": _List(Value("int32")),
        "length": Value("int32"), "native_sr": Value("int32"), "source": Value("string"),
        "category": Value("string"), "augment": Value("int8"),
    })


def load_eval_sets():
    """{name: [(16 kHz float audio, text), ...]} for every saved test set."""
    from datasets import load_from_disk
    root = C.processed_dir() / "tests"
    if not root.exists():
        raise SystemExit(f"No test sets in {root}. Run prepare_data.py first.")
    sets = {}
    for d in sorted(root.iterdir()):
        ds = load_from_disk(str(d)).with_format("numpy", columns=["audio", "text"])
        sets[d.name] = [(np.asarray(r["audio"], np.float32) / 32768.0, str(r["text"])) for r in ds]
    return sets


# =============================================================== training batches
class SpeechCollator:
    """
    Builds one training batch:
      1. 16-bit audio -> float
      2. augment: wide-band IndicVoices clips (augment=1) become phone audio with probability phone_sim_prob,
         a new random version every epoch; real 8 kHz recordings stay as they are; clean clips (augment=2)
         only get a small volume change; dev/test clips (augment=0) are left alone
      3. pad audio to the longest clip (multiple of 320 samples) + attention mask (1 = audio, 0 = padding)
      4. pad labels with -100 (ignored by the loss)
    """

    def __init__(self, phone_sim_prob, seed=0):
        self.phone_sim_prob, self.seed, self.pid, self.rng = phone_sim_prob, seed, None, None

    def __call__(self, feats):
        import torch
        if self.pid != os.getpid():                   # each dataloader worker gets its own random stream
            self.pid = os.getpid()
            self.rng = np.random.default_rng([self.seed, self.pid, time.time_ns() % (2 ** 32)])
        waves = []
        for f in feats:
            y = np.asarray(f["audio"], dtype=np.float32) / 32768.0
            aug = int(f["augment"])
            if aug == 1 and int(f["native_sr"]) > 8000 and self.rng.random() < self.phone_sim_prob:
                y = phone_sim(y, self.rng)
            elif aug in (1, 2):
                y = np.clip(y * self.rng.uniform(0.6, 1.4), -1, 1)
            waves.append(y)
        n = max(len(w) for w in waves)
        n += (-n) % 320
        x = np.zeros((len(waves), n), np.float32)
        am = np.zeros((len(waves), n), np.int64)
        for i, w in enumerate(waves):
            x[i, :len(w)] = w
            am[i, :len(w)] = 1
        L = max(len(f["labels"]) for f in feats)
        lab = np.full((len(feats), L), -100, np.int64)
        for i, f in enumerate(feats):
            lab[i, :len(f["labels"])] = f["labels"]
        return {"input_values": torch.from_numpy(x), "attention_mask": torch.from_numpy(am),
                "labels": torch.from_numpy(lab)}


# =============================================================== evaluation
def transcribe(model, tokenizer, arrays, beams=5, batch_size=16):
    """Production decoding, same as moonshine_base_bn/moonshine_bn.py: beam 5, no_repeat_ngram 3, rep. penalty 1.2."""
    import torch
    dev, prec = device(), precision()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(prec)
    model.eval()
    order = np.argsort([-len(a) for a in arrays])
    out = [None] * len(arrays)
    with torch.no_grad():
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            batch = [arrays[i] for i in idx]
            n = max(len(a) for a in batch)
            n += (-n) % 320
            x = np.zeros((len(batch), n), np.float32)
            am = np.zeros((len(batch), n), np.int64)
            for j, a in enumerate(batch):
                x[j, :len(a)] = a
                am[j, :len(a)] = 1
            max_new = min(int(n / SR * 8) + 16, 190)
            with torch.autocast("cuda", dtype=dtype or torch.float16, enabled=dtype is not None):
                ids = model.generate(torch.from_numpy(x).to(dev), attention_mask=torch.from_numpy(am).to(dev),
                                     max_new_tokens=max_new, num_beams=beams, no_repeat_ngram_size=3,
                                     repetition_penalty=1.2, decoder_start_token_id=START_ID,
                                     eos_token_id=EOS_ID, pad_token_id=PAD_ID)
            for j, i in enumerate(idx):
                out[i] = normalize_text(tokenizer.decode(ids[j], skip_special_tokens=True))
    return out


def score(refs, hyps):
    import jiwer
    pairs = [(r, h if h else "<empty>") for r, h in zip(refs, hyps) if r]
    r, h = zip(*pairs)
    return jiwer.wer(list(r), list(h)), jiwer.cer(list(r), list(h))


def evaluate_sets(model, tokenizer, sets, label, beams=5):
    """Returns (DataFrame with WER/CER per test set, {set name: hypotheses})."""
    import pandas as pd
    rows, hyps_all = [], {}
    for name, data in sets.items():
        t0 = time.time()
        hyps = transcribe(model, tokenizer, [y for y, _ in data], beams=beams)
        w, c = score([t for _, t in data], hyps)
        hyps_all[name] = hyps
        rows.append({"model": label, "test_set": name, "clips": len(data), "WER": round(w, 4),
                     "CER": round(c, 4), "sec": round(time.time() - t0, 1)})
        print(f"  {label:10s} {name:22s} WER {w:.3f}  CER {c:.3f}  ({time.time() - t0:.0f}s)")
    return pd.DataFrame(rows), hyps_all


def transcribe_file(model, tokenizer, path, chunk_sec=20, beams=5):
    """Whole audio file in 20 s chunks, like moonshine_base_bn/test_model.py."""
    import librosa
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    size = int(chunk_sec * SR)
    chunks = [c for c in (y[i:i + size] for i in range(0, len(y), size)) if len(c) >= 1600] or [y]
    return " ".join(h for h in transcribe(model, tokenizer, chunks, beams=beams, batch_size=4) if h)
