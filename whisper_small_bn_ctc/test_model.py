"""
Test Noobbbbb/whisper-small-bn-ctc (Whisper-small encoder + CTC head) on CPU with ../sample-audio

What it does
  * loads the model from ./model on CPU (fully offline)
  * decodes every audio file (mp3/wav/flac/ogg/m4a) -> 16 kHz mono
  * splits long audio into chunks (default 30 s)
  * prints transcript, audio duration, inference time and RTF
    (RTF = processing time / audio duration; < 1.0 means faster than real time)
  * if sample-audio/<name>.txt exists, it is used as the reference -> WER / CER
  * every error is printed AND written with full traceback to outputs/logs/

Usage:
    python test_model.py
    python test_model.py --audio "../sample-audio/ai voice.mp3" --threads 8
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# The model is loaded from local files; never let HF libraries call the network
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL_DIR = HERE / "model"
DEFAULT_AUDIO_DIR = ROOT / "sample-audio"
OUT_DIR = HERE / "outputs"
LOG_DIR = OUT_DIR / "logs"
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus", ".webm"}
SAMPLE_RATE = 16000

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

# ============================ model-specific part ============================ #
MODEL_NAME = "whisper_small_bn_ctc"
DEFAULT_CHUNK = 30.0  # Whisper works on 30 s windows; keep chunks <= 30 s


def add_model_args(parser):
    pass


def load_model(args):
    from whisper_ctc import WhisperCTC
    return WhisperCTC(MODEL_DIR, threads=args.threads)


def model_info(args) -> str:
    return "greedy CTC"
# ============================================================================ #


def setup_logging(verbose: bool) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"test_{RUN_ID}.log"
    logger = logging.getLogger("test")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(fmt)
    file_h = logging.FileHandler(log_file, encoding="utf-8")
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(fmt)
    logger.addHandler(console)
    logger.addHandler(file_h)
    logger.info("Log file: %s", log_file)
    return logger


def log_exception(log: logging.Logger, msg: str, exc: BaseException) -> str:
    log.error("%s: %s: %s", msg, type(exc).__name__, exc)
    log.debug("Traceback:\n%s", traceback.format_exc())
    return f"{type(exc).__name__}: {exc}"


def load_audio(path: Path):
    import librosa
    audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    if audio.size == 0:
        raise ValueError("audio file decoded to 0 samples")
    return audio, len(audio) / SAMPLE_RATE


def split_chunks(audio, chunk_sec: float):
    size = int(chunk_sec * SAMPLE_RATE)
    if size <= 0 or len(audio) <= size:
        return [audio]
    # no tail merging: a chunk must never exceed the model's window (Whisper: 30 s)
    return [audio[i:i + size] for i in range(0, len(audio), size)]


def score(reference: str, hypothesis: str):
    try:
        import jiwer
    except ImportError:
        return None, None
    return jiwer.wer(reference, hypothesis), jiwer.cer(reference, hypothesis)


def collect_audio(arg):
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
                        help="CPU threads (default: all cores)")
    parser.add_argument("--chunk", type=float, default=DEFAULT_CHUNK,
                        help=f"chunk length in seconds for long audio (default {DEFAULT_CHUNK})")
    parser.add_argument("-v", "--verbose", action="store_true", help="show tracebacks on console")
    add_model_args(parser)
    args = parser.parse_args()

    log = setup_logging(args.verbose)
    log.info("=" * 70)
    log.info("Model: %s (%s) | CPU threads: %s | chunk: %ss",
             MODEL_NAME, model_info(args), args.threads, args.chunk)

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
        log.info("Loading model from %s ...", MODEL_DIR)
        t0 = time.time()
        model = load_model(args)
        load_sec = time.time() - t0
        n_params = None
        for attr in ("model", "encoder"):
            m = getattr(model, attr, None)
            if m is not None and hasattr(m, "parameters"):
                n_params = sum(p.numel() for p in m.parameters())
                if attr == "encoder" and hasattr(model, "head"):
                    n_params += sum(p.numel() for p in model.head.parameters())
                break
        log.info("Model loaded in %.1fs%s", load_sec,
                 f" | params={n_params / 1e6:.1f}M" if n_params else "")
    except Exception as e:  # noqa: BLE001
        log_exception(log, "Model load failed", e)
        return 1

    results = []
    for f in files:
        log.info("-" * 70)
        log.info("File: %s", f.name)
        rec = {"file": f.name, "status": "ok"}
        try:
            audio, dur = load_audio(f)
            chunks = split_chunks(audio, args.chunk)
            log.info("Duration: %.2fs | chunks: %d", dur, len(chunks))
            t0 = time.time()
            text = " ".join(t for t in (model.transcribe(c) for c in chunks) if t)
            infer = time.time() - t0
            rec.update(duration_sec=round(dur, 2), chunks=len(chunks), text=text,
                       inference_sec=round(infer, 3), rtf=round(infer / dur, 3))
            log.info("Inference: %.2fs | RTF: %.3f", infer, rec["rtf"])
            log.info("TRANSCRIPT:\n%s", text or "<empty>")

            ref_file = f.with_suffix(".txt")
            if ref_file.exists():
                ref = ref_file.read_text(encoding="utf-8").strip()
                wer, cer = score(ref, text)
                if wer is not None:
                    rec.update(reference=ref, wer=round(wer, 4), cer=round(cer, 4))
                    log.info("WER: %.2f%% | CER: %.2f%%", wer * 100, cer * 100)
        except Exception as e:  # noqa: BLE001
            rec["status"] = "error"
            rec["error"] = log_exception(log, f"Failed on {f.name}", e)
        results.append(rec)

    # ---- save results ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / f"results_{RUN_ID}.json"
    txt_path = OUT_DIR / f"results_{RUN_ID}.txt"
    json_path.write_text(json.dumps({"model": MODEL_NAME, "settings": model_info(args),
                                     "threads": args.threads, "chunk_sec": args.chunk,
                                     "load_sec": round(load_sec, 2), "results": results},
                                    ensure_ascii=False, indent=2), encoding="utf-8")
    with txt_path.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(f"### {r['file']}  [{r['status']}]\n")
            if r["status"] != "ok":
                fh.write(f"ERROR: {r['error']}\n\n")
                continue
            fh.write(f"duration={r['duration_sec']}s  inference={r['inference_sec']}s  rtf={r['rtf']}\n")
            if "wer" in r:
                fh.write(f"WER={r['wer']:.2%}  CER={r['cer']:.2%}\n")
            fh.write(r["text"] + "\n\n")

    ok = [r for r in results if r["status"] == "ok"]
    log.info("=" * 70)
    log.info("SUMMARY: %d ok, %d failed | model load %.1fs", len(ok), len(results) - len(ok), load_sec)
    if ok:
        ta = sum(r["duration_sec"] for r in ok)
        ti = sum(r["inference_sec"] for r in ok)
        log.info("Total audio %.1fs | inference %.1fs | overall RTF %.3f", ta, ti, ti / ta)
    log.info("Results: %s", txt_path)
    log.info("JSON   : %s", json_path)
    return 0 if len(ok) == len(results) else 2


if __name__ == "__main__":
    sys.exit(main())
