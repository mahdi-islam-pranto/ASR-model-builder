"""
Compare the ORIGINAL Moonshine-Base-BN with a FINE-TUNED copy on your own audio, on CPU.

Writes a side-by-side markdown report to Fine-tune-moonshine/outputs/ with, per file:
  * both transcripts
  * RTF (processing time / audio length) for both models
  * WER / CER for both models if a reference transcript <audio name>.txt sits next to the audio

Decoding is identical to moonshine_base_bn/test_model.py (same MoonshineBN class, beam 5,
same 20 s chunks), so the numbers are comparable with the earlier comparison report.

Usage (from the project root):
    python Fine-tune-moonshine/test_finetuned.py --model Fine-tune-moonshine/models/stage1_subakko_phone
    python Fine-tune-moonshine/test_finetuned.py --model <dir> --audio "sample-audio/hospital inbound call center.mp3"
    python Fine-tune-moonshine/test_finetuned.py --model <dir> --beams 1        # faster, greedy
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

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "moonshine_base_bn"))

SR = 16000
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus", ".webm"}

# same text cleaning as the training notebook, so WER here matches WER there
_TAGS = re.compile(r"\[[^\]]*\]|<[^>]*>|\{[^}]*\}")
_PUNCT = re.compile(r"[।॥,\.\?!;:\"'‘’“”\-–—\(\)/\\…*_~`|=+#@&%^]")
_ZW = re.compile(r"[​‌‍﻿]")


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "")
    s = _TAGS.sub(" ", s)
    s = _ZW.sub("", s)
    s = _PUNCT.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def score(ref: str, hyp: str):
    import jiwer
    ref, hyp = normalize_text(ref), normalize_text(hyp) or "<empty>"
    return jiwer.wer(ref, hyp), jiwer.cer(ref, hyp)


def collect_audio(arg: Path):
    if arg.is_file():
        return [arg]
    if not arg.is_dir():
        raise FileNotFoundError(f"Audio path not found: {arg}")
    return sorted(p for p in arg.iterdir() if p.suffix.lower() in AUDIO_EXTS)


def run(asr, audio, chunk_sec):
    size = int(chunk_sec * SR)
    chunks = [audio[i:i + size] for i in range(0, len(audio), size)] or [audio]
    t0 = time.perf_counter()
    text = " ".join(t for t in (asr.transcribe(c) for c in chunks) if t)
    return text, time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description="Original vs fine-tuned Moonshine-Base-BN on CPU")
    ap.add_argument("--model", required=True, help="fine-tuned model folder (unzipped final_model)")
    ap.add_argument("--base", default=str(ROOT / "moonshine_base_bn" / "model"), help="original model folder")
    ap.add_argument("--audio", default=str(ROOT / "sample-audio"), help="audio file or folder")
    ap.add_argument("--chunk", type=float, default=20.0, help="chunk length in seconds (default 20)")
    ap.add_argument("--beams", type=int, default=5, help="beam size (5 = production, 1 = greedy)")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    args = ap.parse_args()

    import librosa
    from moonshine_bn import MoonshineBN

    ft_dir = Path(args.model).resolve()
    info_file = ft_dir / "training_info.json"
    info = json.loads(info_file.read_text(encoding="utf-8")) if info_file.exists() else {}
    run_name = info.get("run_name", ft_dir.name)

    print(f"Loading original model : {args.base}")
    base = MoonshineBN(args.base, threads=args.threads, beams=args.beams)
    print(f"Loading fine-tuned model: {ft_dir}")
    ft = MoonshineBN(ft_dir, threads=args.threads, beams=args.beams)

    files = collect_audio(Path(args.audio))
    if not files:
        print("No audio files found.")
        return 1

    lines = [f"# Original vs fine-tuned Moonshine-Base-BN: `{run_name}`", "",
             f"Generated {datetime.now():%Y-%m-%d %H:%M}. Beam {args.beams}, {args.chunk:.0f} s chunks, "
             f"{args.threads} CPU threads.", ""]
    if info:
        lines += [f"Training run: {info.get('train_hours')} h of training audio, "
                  f"LR {info.get('learning_rate')}, epochs {info.get('epochs')}, "
                  f"best dev WER (greedy) {info.get('best_dev_wer_greedy')}.", ""]
    table = ["| File | Length | RTF original | RTF fine-tuned | WER original | WER fine-tuned | CER original | CER fine-tuned |",
             "|---|---|---|---|---|---|---|---|"]
    details = []

    for p in files:
        audio, _ = librosa.load(str(p), sr=SR, mono=True)
        dur = len(audio) / SR
        print(f"\n== {p.name} ({dur:.1f} s)")
        t_base, s_base = run(base, audio, args.chunk)
        t_ft, s_ft = run(ft, audio, args.chunk)
        print(f"original  : {t_base}")
        print(f"fine-tuned: {t_ft}")
        ref_file = p.with_suffix(".txt")
        w = ["-"] * 4
        if ref_file.exists():
            ref = ref_file.read_text(encoding="utf-8")
            (wb, cb), (wf, cf) = score(ref, t_base), score(ref, t_ft)
            w = [f"{wb:.3f}", f"{wf:.3f}", f"{cb:.3f}", f"{cf:.3f}"]
            print(f"WER {wb:.3f} -> {wf:.3f} | CER {cb:.3f} -> {cf:.3f}")
        table.append(f"| {p.name} | {dur:.1f} s | {s_base / dur:.3f} | {s_ft / dur:.3f} | {w[0]} | {w[1]} | {w[2]} | {w[3]} |")
        details += [f"## {p.name}", "", f"- **original:** {t_base}", f"- **fine-tuned:** {t_ft}", ""]

    lines += table + ["", "WER/CER only where `<audio name>.txt` exists next to the audio file.", ""] + details
    out_dir = HERE / "outputs"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"compare_{run_name}_{datetime.now():%Y%m%d_%H%M%S}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
