"""
Test AI4Bharat IndicConformer-600M (Bengali) on CPU with the audio in ../sample-audio

What it does
  * loads the ONNX model from ./model/<variant> (fully offline, CPU only)
  * decodes every audio file (mp3/wav/flac/ogg/m4a) -> 16 kHz mono
  * splits long audio into chunks (default 30 s) to keep CPU memory low
  * transcribes with CTC (fast), RNNT (usually a bit more accurate, slower) or both
  * prints transcript, audio duration, inference time and RTF
    (RTF = processing time / audio duration; < 1.0 means faster than real time)
  * if sample-audio/<name>.txt exists, it is used as the reference -> WER / CER
  * every error is printed AND written with full traceback to outputs/logs/

Usage:
    python test_model.py                              # int8, CTC + RNNT
    python test_model.py --variant fp32 --decoding ctc
    python test_model.py --audio "../sample-audio/ai voice.mp3" --threads 8 --chunk 20
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

MODEL_NAME = "indic_conformer_600m_bn"

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_AUDIO_DIR = ROOT / "sample-audio"
OUT_DIR = HERE / "outputs"
LOG_DIR = OUT_DIR / "logs"
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus", ".webm"}
SAMPLE_RATE = 16000

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")


# --------------------------------------------------------------------------- #
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
    file_h.setLevel(logging.DEBUG)  # full tracebacks always go to the file
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
    chunks = [audio[i:i + size] for i in range(0, len(audio), size)]
    if len(chunks) > 1 and len(chunks[-1]) < SAMPLE_RATE:  # merge tiny (<1 s) tail
        import numpy as np
        chunks[-2] = np.concatenate([chunks[-2], chunks.pop()])
    return chunks


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


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=f"Test {MODEL_NAME} on CPU")
    parser.add_argument("--variant", choices=["int8", "fp32"], default="int8",
                        help="which download to use (default int8)")
    parser.add_argument("--decoding", choices=["ctc", "rnnt", "both"], default="both")
    parser.add_argument("--audio", help="audio file or folder (default: ../sample-audio)")
    parser.add_argument("--threads", type=int, default=os.cpu_count(),
                        help="CPU threads (default: all cores)")
    parser.add_argument("--chunk", type=float, default=30.0,
                        help="chunk length in seconds for long audio (0 = no chunking)")
    parser.add_argument("-v", "--verbose", action="store_true", help="show tracebacks on console")
    args = parser.parse_args()

    log = setup_logging(args.verbose)
    model_dir = HERE / "model" / args.variant
    decodings = ["ctc", "rnnt"] if args.decoding == "both" else [args.decoding]
    log.info("=" * 70)
    log.info("Model: %s (%s) | decoding: %s | CPU threads: %s | chunk: %ss",
             MODEL_NAME, args.variant, "+".join(decodings), args.threads, args.chunk)

    try:
        import onnxruntime as ort
        import torch
        torch.set_num_threads(args.threads)
        log.info("onnxruntime %s | torch %s | python %s",
                 ort.__version__, torch.__version__, sys.version.split()[0])
    except Exception as e:  # noqa: BLE001
        log_exception(log, "Missing dependency (pip install -r ../requirements.txt)", e)
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
        from indic_conformer import IndicConformerONNX
        log.info("Loading model from %s ...", model_dir)
        t0 = time.time()
        model = IndicConformerONNX(model_dir, lang="bn", threads=args.threads,
                                   load_rnnt="rnnt" in decodings)
        load_sec = time.time() - t0
        log.info("Model loaded in %.1fs", load_sec)
    except Exception as e:  # noqa: BLE001
        log_exception(log, "Model load failed", e)
        if not model_dir.exists():
            log.error("Run first: python download_model.py --variant %s", args.variant)
        return 1

    results = []
    for f in files:
        log.info("-" * 70)
        log.info("File: %s", f.name)
        rec = {"file": f.name, "status": "ok"}
        try:
            audio, dur = load_audio(f)
            chunks = split_chunks(audio, args.chunk)
            rec.update(duration_sec=round(dur, 2), chunks=len(chunks))
            log.info("Duration: %.2fs | chunks: %d", dur, len(chunks))

            ref_file = f.with_suffix(".txt")
            ref = ref_file.read_text(encoding="utf-8").strip() if ref_file.exists() else None
            if ref:
                rec["reference"] = ref

            for dec in decodings:
                t0 = time.time()
                text = " ".join(t for t in (model.transcribe(c, dec) for c in chunks) if t)
                infer = time.time() - t0
                r = {"text": text, "inference_sec": round(infer, 3), "rtf": round(infer / dur, 3)}
                log.info("[%s] Inference: %.2fs | RTF: %.3f", dec.upper(), infer, r["rtf"])
                log.info("[%s] TRANSCRIPT:\n%s", dec.upper(), text if text else "<empty>")
                if ref:
                    wer, cer = score(ref, text)
                    if wer is not None:
                        r.update(wer=round(wer, 4), cer=round(cer, 4))
                        log.info("[%s] WER: %.2f%% | CER: %.2f%%", dec.upper(), wer * 100, cer * 100)
                rec[dec] = r
        except Exception as e:  # noqa: BLE001
            rec["status"] = "error"
            rec["error"] = log_exception(log, f"Failed on {f.name}", e)
        results.append(rec)

    # ---- save results ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.variant}_{RUN_ID}"
    json_path = OUT_DIR / f"results_{tag}.json"
    txt_path = OUT_DIR / f"results_{tag}.txt"
    json_path.write_text(json.dumps({"model": MODEL_NAME, "variant": args.variant,
                                     "threads": args.threads, "chunk_sec": args.chunk,
                                     "load_sec": round(load_sec, 2), "results": results},
                                    ensure_ascii=False, indent=2), encoding="utf-8")
    with txt_path.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(f"### {r['file']}  [{r['status']}]\n")
            if r["status"] != "ok":
                fh.write(f"ERROR: {r['error']}\n\n")
                continue
            fh.write(f"duration={r['duration_sec']}s\n")
            for dec in decodings:
                d = r[dec]
                fh.write(f"[{dec.upper()}] inference={d['inference_sec']}s  rtf={d['rtf']}")
                if "wer" in d:
                    fh.write(f"  WER={d['wer']:.2%}  CER={d['cer']:.2%}")
                fh.write("\n" + d["text"] + "\n")
            fh.write("\n")

    ok = [r for r in results if r["status"] == "ok"]
    log.info("=" * 70)
    log.info("SUMMARY: %d ok, %d failed | model load %.1fs", len(ok), len(results) - len(ok), load_sec)
    total_audio = sum(r["duration_sec"] for r in ok)
    for dec in decodings:
        total_inf = sum(r[dec]["inference_sec"] for r in ok)
        if total_audio:
            log.info("[%s] total audio %.1fs | inference %.1fs | overall RTF %.3f",
                     dec.upper(), total_audio, total_inf, total_inf / total_audio)
    log.info("Results: %s", txt_path)
    log.info("JSON   : %s", json_path)
    return 0 if len(ok) == len(results) else 2


if __name__ == "__main__":
    sys.exit(main())
