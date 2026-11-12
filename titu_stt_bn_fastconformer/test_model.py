"""
Test hishab/titu_stt_bn_fastconformer on CPU with the audio in ../sample-audio

What it does
  * loads the .nemo model from ./model on CPU
  * decodes every audio file (mp3/wav/flac/ogg/m4a) -> 16 kHz mono
  * splits long audio into chunks (default 30 s) so CPU memory stays low
  * prints the transcript, audio duration, inference time and RTF
    (RTF = processing time / audio duration; < 1.0 means faster than real time)
  * if sample-audio/<name>.txt exists, it is used as the reference
    transcript and WER / CER are computed
  * every error is printed AND written with full traceback to outputs/logs/

Usage:
    python test_model.py
    python test_model.py --audio "../sample-audio/ai voice.mp3"
    python test_model.py --threads 8 --chunk 20
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path

MODEL_NAME = "titu_stt_bn_fastconformer"
MODEL_FILE = "titu_stt_bn_fastconformer.nemo"

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL_PATH = HERE / "model" / MODEL_FILE
DEFAULT_AUDIO_DIR = ROOT / "sample-audio"
OUT_DIR = HERE / "outputs"
LOG_DIR = OUT_DIR / "logs"
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus", ".webm"}
SAMPLE_RATE = 16000

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

# The model is loaded from a local file, so no network is needed. Without these,
# NeMo / Hugging Face libraries make network calls on import and can hang for
# many minutes when the internet is slow or down (important for an offline server).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #
def setup_logging(verbose: bool) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"test_{RUN_ID}.log"
    logger = logging.getLogger("test")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(fmt)
    file_h = logging.FileHandler(log_file, encoding="utf-8")
    file_h.setLevel(logging.DEBUG)  # full tracebacks always go to the file
    file_h.setFormatter(fmt)
    logger.addHandler(console)
    logger.addHandler(file_h)
    logger.propagate = False

    logger.info("Log file: %s", log_file)
    return logger


def log_exception(log: logging.Logger, msg: str, exc: BaseException) -> str:
    """Print a short error to console and the full traceback to the log file."""
    tb = traceback.format_exc()
    log.error("%s: %s: %s", msg, type(exc).__name__, exc)
    log.debug("Traceback:\n%s", tb)
    return f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# audio
# --------------------------------------------------------------------------- #
def load_audio(path: Path):
    """Return (float32 mono numpy array @16 kHz, duration_seconds)."""
    import librosa

    audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    if audio.size == 0:
        raise ValueError("audio file decoded to 0 samples")
    return audio, len(audio) / SAMPLE_RATE


def split_chunks(audio, chunk_sec: float):
    size = int(chunk_sec * SAMPLE_RATE)
    if size <= 0 or len(audio) <= size:
        return [audio]
    chunks = [audio[i:i + size] for i in range(0, len(audio), size)]
    # merge a tiny tail (< 1 s) into the previous chunk
    if len(chunks) > 1 and len(chunks[-1]) < SAMPLE_RATE:
        import numpy as np
        chunks[-2] = np.concatenate([chunks[-2], chunks[-1]])
        chunks.pop()
    return chunks


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def load_model(log: logging.Logger):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run: python download_model.py"
        )
    import nemo.collections.asr as nemo_asr

    quiet = log.handlers[0].level != logging.DEBUG  # keep NeMo warnings when -v is used
    if quiet:
        quiet_nemo(log)
    log.info("Loading model %s (CPU) ...", MODEL_PATH.name)
    t0 = time.time()
    model = nemo_asr.models.ASRModel.restore_from(str(MODEL_PATH), map_location="cpu")
    if quiet:
        quiet_nemo(log)  # restore_from resets NeMo's log level
    model.eval()
    model = model.to("cpu")
    log.info("Model loaded in %.1fs  | class=%s | params=%.1fM",
             time.time() - t0, type(model).__name__,
             sum(p.numel() for p in model.parameters()) / 1e6)

    # NeMo recommends 'greedy_batch' for CTC: same output, faster
    try:
        from omegaconf import open_dict
        cfg = model.cfg.decoding
        if cfg.get("strategy") == "greedy":
            with open_dict(cfg):
                cfg.strategy = "greedy_batch"
            model.change_decoding_strategy(cfg, verbose=False)
            log.info("Decoding strategy: greedy_batch")
    except Exception as e:  # noqa: BLE001 - non-fatal, keep default decoding
        log.warning("Could not switch decoding to greedy_batch: %s", e)
    return model


def quiet_nemo(log: logging.Logger) -> None:
    """Hide NeMo's harmless config warnings (use -v to see them). NeMo errors still show."""
    try:
        import warnings
        from nemo.utils import logging as nemo_logging
        nemo_logging.setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", message=".*pin_memory.*")
    except Exception as e:  # noqa: BLE001
        log.debug("Could not lower NeMo log level: %s", e)


def hyp_to_text(h) -> str:
    """NeMo returns str, Hypothesis, or nested lists depending on version/model."""
    if isinstance(h, str):
        return h
    if hasattr(h, "text"):
        return h.text or ""
    if isinstance(h, (list, tuple)) and h:
        return hyp_to_text(h[0])
    return str(h)


def transcribe_chunks(model, chunks, tmp_dir: Path, batch_size: int) -> str:
    """Write chunks to temp wavs (most NeMo-version-proof input) and transcribe."""
    import soundfile as sf
    import torch

    paths = []
    for i, c in enumerate(chunks):
        p = tmp_dir / f"chunk_{i:04d}.wav"
        sf.write(str(p), c, SAMPLE_RATE, subtype="PCM_16")
        paths.append(str(p))

    with torch.inference_mode():
        out = model.transcribe(paths, batch_size=batch_size, verbose=False)
    # hybrid models return (best_hyps, all_hyps)
    if isinstance(out, tuple):
        out = out[0]
    texts = [hyp_to_text(h).strip() for h in out]
    return " ".join(t for t in texts if t)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def score(reference: str, hypothesis: str):
    try:
        import jiwer
    except ImportError:
        return None, None
    return jiwer.wer(reference, hypothesis), jiwer.cer(reference, hypothesis)


# --------------------------------------------------------------------------- #
def collect_audio(arg: str | None) -> list[Path]:
    target = Path(arg) if arg else DEFAULT_AUDIO_DIR
    if target.is_file():
        return [target]
    if not target.is_dir():
        raise FileNotFoundError(f"Audio path not found: {target}")
    return sorted(p for p in target.iterdir() if p.suffix.lower() in AUDIO_EXTS)


def main() -> int:
    parser = argparse.ArgumentParser(description=f"Test {MODEL_NAME} on CPU")
    parser.add_argument("--audio", help="audio file or folder (default: ../sample-audio)")
    parser.add_argument("--threads", type=int, default=os.cpu_count(),
                        help="CPU threads for torch (default: all cores)")
    parser.add_argument("--chunk", type=float, default=30.0,
                        help="chunk length in seconds for long audio (0 = no chunking)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("-v", "--verbose", action="store_true", help="show debug logs/tracebacks on console")
    args = parser.parse_args()

    log = setup_logging(args.verbose)
    log.info("=" * 70)
    log.info("Model: %s | device: CPU | threads: %s | chunk: %ss", MODEL_NAME, args.threads, args.chunk)

    try:
        import torch
        torch.set_num_threads(args.threads)
        log.info("torch %s | python %s", torch.__version__, sys.version.split()[0])
    except Exception as e:  # noqa: BLE001
        log_exception(log, "Failed to import torch", e)
        return 1

    try:
        files = collect_audio(args.audio)
    except Exception as e:  # noqa: BLE001
        log_exception(log, "Could not find audio", e)
        return 1
    if not files:
        log.error("No audio files found in %s", args.audio or DEFAULT_AUDIO_DIR)
        return 1
    log.info("Found %d audio file(s)", len(files))

    try:
        t_load = time.time()
        model = load_model(log)
        load_sec = time.time() - t_load
    except Exception as e:  # noqa: BLE001
        log_exception(log, "Model load failed", e)
        log.error("Full traceback in %s", LOG_DIR)
        return 1

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for f in files:
            log.info("-" * 70)
            log.info("File: %s", f.name)
            rec = {"file": f.name, "status": "ok"}
            try:
                t0 = time.time()
                audio, dur = load_audio(f)
                rec["load_audio_sec"] = round(time.time() - t0, 3)
                rec["duration_sec"] = round(dur, 2)
                chunks = split_chunks(audio, args.chunk)
                log.info("Duration: %.2fs | chunks: %d", dur, len(chunks))

                t0 = time.time()
                text = transcribe_chunks(model, chunks, tmp_dir, args.batch_size)
                infer = time.time() - t0
                for p in tmp_dir.glob("chunk_*.wav"):
                    p.unlink()

                rec.update(text=text, inference_sec=round(infer, 3),
                           rtf=round(infer / dur, 3) if dur else None)
                log.info("Inference: %.2fs | RTF: %.3f", infer, rec["rtf"])
                log.info("TRANSCRIPT:\n%s", text if text else "<empty>")

                ref_file = f.with_suffix(".txt")
                if ref_file.exists():
                    ref = ref_file.read_text(encoding="utf-8").strip()
                    wer, cer = score(ref, text)
                    if wer is not None:
                        rec.update(reference=ref, wer=round(wer, 4), cer=round(cer, 4))
                        log.info("WER: %.2f%% | CER: %.2f%%", wer * 100, cer * 100)
                    else:
                        log.warning("jiwer not installed; skipping WER/CER")
            except Exception as e:  # noqa: BLE001
                rec["status"] = "error"
                rec["error"] = log_exception(log, f"Failed on {f.name}", e)
            results.append(rec)

    # ---- save results ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / f"results_{RUN_ID}.json"
    txt_path = OUT_DIR / f"results_{RUN_ID}.txt"
    json_path.write_text(json.dumps({"model": MODEL_NAME, "threads": args.threads, "load_sec": round(load_sec, 2),
                                     "chunk_sec": args.chunk, "results": results},
                                    ensure_ascii=False, indent=2), encoding="utf-8")
    with txt_path.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(f"### {r['file']}  [{r['status']}]\n")
            if r["status"] == "ok":
                fh.write(f"duration={r['duration_sec']}s  inference={r['inference_sec']}s  rtf={r['rtf']}\n")
                if "wer" in r:
                    fh.write(f"WER={r['wer']:.2%}  CER={r['cer']:.2%}\n")
                fh.write(r["text"] + "\n\n")
            else:
                fh.write(f"ERROR: {r['error']}\n\n")

    ok = [r for r in results if r["status"] == "ok"]
    log.info("=" * 70)
    log.info("SUMMARY: %d ok, %d failed", len(ok), len(results) - len(ok))
    if ok:
        total_audio = sum(r["duration_sec"] for r in ok)
        total_inf = sum(r["inference_sec"] for r in ok)
        log.info("Total audio %.1fs | total inference %.1fs | overall RTF %.3f",
                 total_audio, total_inf, total_inf / total_audio if total_audio else 0)
    log.info("Results: %s", txt_path)
    log.info("JSON   : %s", json_path)
    return 0 if len(ok) == len(results) else 2


if __name__ == "__main__":
    sys.exit(main())
