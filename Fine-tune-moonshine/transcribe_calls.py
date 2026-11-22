"""
Transcribe call recordings of ANY length with the fine-tuned Moonshine model (CPU is fine).

    python Fine-tune-moonshine/transcribe_calls.py                       # all audio in sample-audio/
    python Fine-tune-moonshine/transcribe_calls.py --audio "D:/calls"    # a folder (or one file)
    python Fine-tune-moonshine/transcribe_calls.py --compare             # also the ORIGINAL model, side by side
    python Fine-tune-moonshine/transcribe_calls.py --split-channels      # stereo calls: agent/caller per channel

What it does with every file:
  1. load + convert to 16 kHz mono (8 kHz call recordings are resampled automatically; the model only accepts
     16 kHz). Any format librosa can read: wav, mp3, m4a, ogg, flac, ...
  2. cut it into pieces of at most 20 s AT PAUSES (silence), never in the middle of a word. The model was
     trained on clips up to 20 s and its decoder has only 194 positions, so one long piece would be cut off.
  3. transcribe every piece (beam 5, same settings as the evaluation), join the text
Output in Fine-tune-moonshine/outputs/transcripts_<time>/:
  <file>.txt           the full transcript
  <file>.segments.txt  the pieces with start/end times ([00:12.3 - 00:27.9] ...)
  report.md            all files: length, speed (RTF), transcripts, WER/CER if <file>.txt exists next to the audio
"""

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "moonshine_base_bn"))

SR = 16000
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus", ".webm", ".aac", ".wma"}

# same text cleaning as training/evaluation, so WER here is comparable
_TAGS = re.compile(r"\[[^\]]*\]|<[^>]*>|\{[^}]*\}")
_PUNCT = re.compile(r"[।॥,\.\?!;:\"'‘’“”\-–—\(\)/\\…*_~`|=+#@&%^]")
_ZW = re.compile(r"[​‌‍﻿]")


def normalize_text(s):
    s = unicodedata.normalize("NFC", s or "")
    s = _TAGS.sub(" ", s)
    s = _ZW.sub("", s)
    s = _PUNCT.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def score(ref, hyp):
    import jiwer
    ref, hyp = normalize_text(ref), normalize_text(hyp) or "<empty>"
    return jiwer.wer(ref, hyp), jiwer.cer(ref, hyp)


# ------------------------------------------------------------------ cutting long audio at pauses
def split_at_pauses(y, max_sec=20.0, top_db=35, pad_sec=0.15):
    """
    Returns [(start_sample, end_sample)] pieces of at most max_sec, cut in silences.
      * librosa.effects.split finds the parts with sound (quieter than the loudest part by more than top_db
        dB = silence). Silences are where we may cut.
      * neighbouring sound parts are joined until the piece would be longer than max_sec
      * one sound part longer than max_sec (someone talks 25 s without a pause) is cut at its quietest
        moment in the last 5 s before the limit
    """
    import librosa
    pad = int(pad_sec * SR)
    max_len = int(max_sec * SR) - 2 * pad                 # room for the padding added at the end
    voiced = librosa.effects.split(y, top_db=top_db, frame_length=1024, hop_length=256)
    parts = []
    for s, e in voiced:                                   # make every sound part <= max_len
        while e - s > max_len:
            cut = quietest_point(y, s + max_len - 5 * SR, s + max_len)
            parts.append((s, cut))
            s = cut
        parts.append((s, e))
    pieces, cur = [], None
    for s, e in parts:                                    # join parts into pieces <= max_len
        if cur and e - cur[0] <= max_len:
            cur[1] = e
        else:
            if cur:
                pieces.append(tuple(cur))
            cur = [s, e]
    if cur:
        pieces.append(tuple(cur))
    out = []
    for s, e in pieces:                                   # a little context around each piece
        s, e = max(0, s - pad), min(len(y), e + pad)
        if e - s >= int(0.3 * SR):
            out.append((s, e))
    return out


def quietest_point(y, a, b, win=int(0.05 * SR)):
    a = max(a, 0)
    seg = y[a:b]
    if len(seg) <= win:
        return b
    energy = np.convolve(seg ** 2, np.ones(win) / win, mode="valid")
    return a + int(np.argmin(energy)) + win // 2


def ts(sample):
    t = sample / SR
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def transcribe(asr, y, max_sec):
    """-> (full text, [(start, end, text)], seconds taken)"""
    t0 = time.perf_counter()
    segs = []
    for s, e in split_at_pauses(y, max_sec=max_sec):
        text = normalize_text(asr.transcribe(y[s:e]))
        if text:
            segs.append((s, e, text))
    return " ".join(t for _, _, t in segs), segs, time.perf_counter() - t0


def load_channels(path, split):
    """[(label, 16 kHz float audio)]: one mono mix, or one entry per channel with --split-channels."""
    import librosa
    y, _ = librosa.load(str(path), sr=SR, mono=False)     # resamples 8 kHz (or anything) to 16 kHz
    if y.ndim == 1:
        return [("", y)]
    if split:
        return [(f"ch{i + 1}", ch) for i, ch in enumerate(y)]
    return [("", y.mean(axis=0))]


def collect_audio(arg):
    if arg.is_file():
        return [arg]
    if not arg.is_dir():
        raise SystemExit(f"Audio path not found: {arg}")
    return sorted(p for p in arg.rglob("*") if p.suffix.lower() in AUDIO_EXTS)


def run_model(asr, channels, max_sec):
    """Transcribe all channels; segments of all channels sorted by start time."""
    all_segs, secs = [], 0.0
    for label, y in channels:
        _, segs, t = transcribe(asr, y, max_sec)
        secs += t
        all_segs += [(s, e, label, text) for s, e, text in segs]
    all_segs.sort()
    return " ".join(t for *_, t in all_segs), all_segs, secs


def seg_lines(segs):
    return [f"[{ts(s)} - {ts(e)}]{' ' + lab if lab else ''} {t}" for s, e, lab, t in segs]


def main():
    ap = argparse.ArgumentParser(description="Transcribe call recordings of any length with the fine-tuned model")
    ap.add_argument("--audio", default=str(ROOT / "sample-audio"), help="audio file or folder (searched recursively)")
    ap.add_argument("--model", default=str(HERE / "model"), help="fine-tuned model folder")
    ap.add_argument("--compare", action="store_true", help="also run the original model, side by side")
    ap.add_argument("--base", default=str(ROOT / "moonshine_base_bn" / "model"), help="original model folder")
    ap.add_argument("--beams", type=int, default=5, help="5 = best quality (default), 1 = ~3x faster")
    ap.add_argument("--max-sec", type=float, default=20.0, help="longest piece in seconds (<= 20)")
    ap.add_argument("--split-channels", action="store_true", help="stereo: transcribe each channel separately")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    args = ap.parse_args()
    if args.max_sec > 20:
        raise SystemExit("--max-sec must be <= 20 (the model was trained on clips up to 20 s)")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    from moonshine_bn import MoonshineBN

    files = collect_audio(Path(args.audio))
    if not files:
        raise SystemExit(f"No audio files in {args.audio}")
    model_dir = Path(args.model).resolve()
    info_file = model_dir / "training_info.json"
    run_name = json.loads(info_file.read_text(encoding="utf-8")).get("run_name", model_dir.name) \
        if info_file.exists() else model_dir.name

    print(f"fine-tuned model: {model_dir}")
    models = {"fine-tuned": MoonshineBN(model_dir, threads=args.threads, beams=args.beams)}
    if args.compare:
        print(f"original model  : {args.base}")
        models["original"] = MoonshineBN(args.base, threads=args.threads, beams=args.beams)

    out_dir = HERE / "outputs" / f"transcripts_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(models)
    md = [f"# Transcripts: `{run_name}`", "",
          f"{datetime.now():%Y-%m-%d %H:%M}, beam {args.beams}, pieces <= {args.max_sec:.0f} s cut at pauses, "
          f"{args.threads} CPU threads. RTF = processing time / audio length (< 1 = faster than real time).", "",
          "| File | Length | " + " | ".join(f"RTF {n}" for n in names) + " | "
          + " | ".join(f"WER {n}" for n in names) + " | " + " | ".join(f"CER {n}" for n in names) + " |",
          "|---|---|" + "---|" * (3 * len(names))]
    details, totals = [], {n: [0.0, []] for n in names}

    for p in files:
        try:
            channels = load_channels(p, args.split_channels)
        except Exception as e:
            print(f"\n!! {p.name}: cannot read ({e})")
            continue
        dur = max(len(y) for _, y in channels) / SR
        print(f"\n== {p.name} ({dur:.1f} s, {len(channels)} channel(s))")
        ref_file = p.with_suffix(".txt")
        ref = ref_file.read_text(encoding="utf-8") if ref_file.exists() else None
        rtf, wer, cer = [], [], []
        details += [f"## {p.name} ({dur:.1f} s)", ""]
        for n, asr in models.items():
            text, segs, secs = run_model(asr, channels, args.max_sec)
            totals[n][0] += secs
            rtf.append(f"{secs / dur:.3f}")
            print(f"--- {n} ({secs:.1f} s, {len(segs)} pieces)")
            for line in seg_lines(segs):
                print("   ", line)
            stem = p.stem + ("" if n == "fine-tuned" else f".{n}")
            (out_dir / f"{stem}.txt").write_text(text + "\n", encoding="utf-8")
            (out_dir / f"{stem}.segments.txt").write_text("\n".join(seg_lines(segs)) + "\n", encoding="utf-8")
            details += [f"**{n}:**", "", "```", *seg_lines(segs), "```", ""]
            if ref is not None:
                w, c = score(ref, text)
                wer.append(f"{w:.3f}")
                cer.append(f"{c:.3f}")
                totals[n][1].append((normalize_text(ref), text))
                print(f"    WER {w:.3f}  CER {c:.3f}")
            else:
                wer.append("-")
                cer.append("-")
        md.append(f"| {p.name} | {dur:.1f} s | " + " | ".join(rtf) + " | " + " | ".join(wer) + " | "
                  + " | ".join(cer) + " |")

    md += ["", "WER/CER only where `<audio name>.txt` (the correct transcript) is next to the audio file.", ""]
    for n in names:
        pairs = totals[n][1]
        if pairs:
            import jiwer
            refs, hyps = zip(*pairs)
            hyps = [h or "<empty>" for h in hyps]
            line = f"**{n}, all files with a reference:** WER {jiwer.wer(list(refs), hyps):.3f}, " \
                   f"CER {jiwer.cer(list(refs), hyps):.3f} ({len(pairs)} files)"
            md.append(line)
            print(line.replace("**", ""))
    (out_dir / "report.md").write_text("\n".join(md + [""] + details), encoding="utf-8")
    print(f"\nTranscripts + report: {out_dir}")


if __name__ == "__main__":
    main()
