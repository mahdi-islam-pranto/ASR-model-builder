"""
Test the Vosk Bengali streaming Zipformer (sherpa-onnx) on CPU with ../sample-audio

This is a STREAMING (real-time) model. The script simulates a live call:
  * audio is decoded to 16 kHz mono and fed in small pieces (default 100 ms)
  * the recognizer decodes as audio arrives, exactly like on a live server
  * built-in endpoint detection splits speech into utterances on pauses,
    so you get timestamped segments (useful for call recordings)
  * prints transcript, segments, inference time and RTF
    (RTF = processing time / audio duration; < 1.0 means faster than real time)
  * if sample-audio/<name>.txt exists -> WER / CER
  * every error is printed AND written with full traceback to outputs/logs/

Usage:
    python test_model.py
    python test_model.py --decoding greedy_search          # faster, slightly less accurate
    python test_model.py --audio "../sample-audio/ai voice.mp3" --threads 2
    python test_model.py --no-endpoint                     # one block of text, no segments
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

MODEL_NAME = "sherpa_zipformer_bn_streaming"

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL_DIR = HERE / "model"
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
def build_recognizer(args):
    import sherpa_onnx

    am, lang = MODEL_DIR / "am-onnx", MODEL_DIR / "lang"
    for f in (am / "encoder.onnx", am / "decoder.onnx", am / "joiner.onnx", lang / "tokens.txt"):
        if not f.exists():
            raise FileNotFoundError(f"{f} not found. Run: python download_model.py")
    kwargs = dict(
        encoder=str(am / "encoder.onnx"),
        decoder=str(am / "decoder.onnx"),
        joiner=str(am / "joiner.onnx"),
        tokens=str(lang / "tokens.txt"),
        num_threads=args.threads,
        sample_rate=SAMPLE_RATE,
        feature_dim=80,
        decoding_method=args.decoding,
        max_active_paths=args.beam,
        provider="cpu",
        model_type="zipformer2",
        enable_endpoint_detection=not args.no_endpoint,
        # endpoint rules (seconds): silence after no speech / after speech / max utterance
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=args.pause,
        rule3_min_utterance_length=20.0,
    )
    try:
        return sherpa_onnx.OnlineRecognizer.from_transducer(dither=3e-5, **kwargs)
    except TypeError:  # older sherpa-onnx without the dither option
        return sherpa_onnx.OnlineRecognizer.from_transducer(**kwargs)


def result_text(recognizer, stream) -> str:
    r = recognizer.get_result(stream)
    return (r if isinstance(r, str) else getattr(r, "text", str(r))).strip()


def stream_file(recognizer, audio, piece_sec: float):
    """Feed audio piece by piece like a live call. Returns list of (start, end, text)."""
    import numpy as np

    stream = recognizer.create_stream()
    piece = max(1, int(piece_sec * SAMPLE_RATE))
    segments, seg_start, fed = [], 0.0, 0

    def flush(end_time):
        nonlocal stream, seg_start
        text = result_text(recognizer, stream)
        if text:
            segments.append((round(seg_start, 2), round(end_time, 2), text))
        recognizer.reset(stream)
        seg_start = end_time

    for i in range(0, len(audio), piece):
        stream.accept_waveform(SAMPLE_RATE, audio[i:i + piece])
        fed += len(audio[i:i + piece])
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        if recognizer.is_endpoint(stream):
            flush(fed / SAMPLE_RATE)

    # tail padding so the last words are emitted
    stream.accept_waveform(SAMPLE_RATE, np.zeros(int(0.66 * SAMPLE_RATE), dtype=np.float32))
    stream.input_finished()
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    flush(len(audio) / SAMPLE_RATE)
    return segments


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=f"Test {MODEL_NAME} on CPU")
    parser.add_argument("--audio", help="audio file or folder (default: ../sample-audio)")
    parser.add_argument("--decoding", choices=["modified_beam_search", "greedy_search"],
                        default="modified_beam_search")
    parser.add_argument("--beam", type=int, default=4, help="beam size for modified_beam_search")
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1),
                        help="CPU threads per stream (default 4; a live server runs many streams)")
    parser.add_argument("--piece", type=float, default=0.1,
                        help="seconds of audio fed per step (simulated live stream)")
    parser.add_argument("--pause", type=float, default=0.8,
                        help="silence (s) after speech that ends an utterance")
    parser.add_argument("--no-endpoint", action="store_true", help="disable utterance splitting")
    parser.add_argument("-v", "--verbose", action="store_true", help="show tracebacks on console")
    args = parser.parse_args()

    log = setup_logging(args.verbose)
    log.info("=" * 70)
    log.info("Model: %s | decoding: %s | CPU threads: %s | piece: %ss | endpoint: %s",
             MODEL_NAME, args.decoding, args.threads, args.piece, not args.no_endpoint)

    try:
        import sherpa_onnx
        log.info("sherpa-onnx %s | python %s", sherpa_onnx.__version__, sys.version.split()[0])
    except Exception as e:  # noqa: BLE001
        log_exception(log, "sherpa-onnx missing (pip install sherpa-onnx)", e)
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
        t0 = time.time()
        recognizer = build_recognizer(args)
        load_sec = time.time() - t0
        log.info("Model loaded in %.1fs", load_sec)
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
            log.info("Duration: %.2fs", dur)
            t0 = time.time()
            segments = stream_file(recognizer, audio, args.piece)
            infer = time.time() - t0
            text = " ".join(s[2] for s in segments)
            rec.update(duration_sec=round(dur, 2), inference_sec=round(infer, 3),
                       rtf=round(infer / dur, 3), text=text,
                       segments=[{"start": s, "end": e, "text": t} for s, e, t in segments])
            log.info("Inference: %.2fs | RTF: %.3f | segments: %d", infer, rec["rtf"], len(segments))
            log.info("SEGMENTS:\n%s", "\n".join(f"  [{s:7.2f} - {e:7.2f}] {t}" for s, e, t in segments)
                     or "  <none>")
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
    json_path.write_text(json.dumps({"model": MODEL_NAME, "decoding": args.decoding,
                                     "threads": args.threads, "load_sec": round(load_sec, 2),
                                     "results": results}, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    with txt_path.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(f"### {r['file']}  [{r['status']}]\n")
            if r["status"] != "ok":
                fh.write(f"ERROR: {r['error']}\n\n")
                continue
            fh.write(f"duration={r['duration_sec']}s  inference={r['inference_sec']}s  rtf={r['rtf']}\n")
            if "wer" in r:
                fh.write(f"WER={r['wer']:.2%}  CER={r['cer']:.2%}\n")
            for s in r["segments"]:
                fh.write(f"[{s['start']:7.2f} - {s['end']:7.2f}] {s['text']}\n")
            fh.write("\n")

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
