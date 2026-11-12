"""
Compare all tested ASR models in one report.

Reads the LATEST results_*.json from every <model>/outputs/ folder, plus any Colab
result JSONs saved in notebooks/outputs/, and writes comparison/comparison_<time>.md:

  1. summary table: license, size, load time, overall speed (RTF) per model/decoding
  2. per audio file: transcript of every model side by side, with
       - WER / CER  (only if sample-audio/<name>.txt reference exists; computed the
         same way for every model so the numbers are comparable)
       - words   : number of words recognized (more is not always better, see next)
       - repeat% : share of words that are repeats of the single most common word;
                   high values (> 20%) usually mean the model is looping/hallucinating

Usage:
    python compare_models.py
    python compare_models.py --no-colab        # local runs only
"""

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AUDIO_DIR = ROOT / "sample-audio"
OUT_DIR = ROOT / "comparison"

# static facts (from model cards); edit when adding a model
MODEL_INFO = {
    "titu_stt_bn_conformer_large": dict(name="Hishab Conformer-Large", params="121.5M",
                                        license="CC-BY-NC-4.0 (non-commercial)",
                                        streaming="no", finetune="yes (NeMo)"),
    "titu_stt_bn_fastconformer": dict(name="Hishab FastConformer", params="115.6M",
                                      license="CC-BY-NC-4.0 (non-commercial)",
                                      streaming="no", finetune="yes (NeMo)"),
    "indic_conformer_600m_bn": dict(name="AI4Bharat IndicConformer-600M", params="600M",
                                    license="MIT", streaming="no",
                                    finetune="yes (needs NeMo checkpoint, not the ONNX)"),
    "sherpa_zipformer_bn_streaming": dict(name="Vosk Zipformer (streaming)", params="94 MB files",
                                          license="Apache-2.0", streaming="**yes**",
                                          finetune="no (ONNX only; retrain with icefall)"),
    "moonshine_base_bn": dict(name="Moonshine-Base-BN", params="61.5M", license="MIT",
                              streaming="no", finetune="yes (HF transformers)"),
    "whisper_small_bn_ctc": dict(name="Whisper-small-bn CTC", params="102.4M", license="Apache-2.0",
                                 streaming="no", finetune="yes (HF / PyTorch)"),
}


# --------------------------------------------------------------------------- #
def latest_json(folder: Path):
    """Most complete run (most ok files, then most decodings), newest on ties.
    So a quick single-file / single-decoding check does not hide a full run."""
    def rank(p: Path):
        try:
            res = json.loads(p.read_text(encoding="utf-8")).get("results", [])
        except Exception:  # noqa: BLE001 - unreadable file ranks last
            return (-1, -1, 0)
        ok = [r for r in res if r.get("status") == "ok"]
        decodings = sum(1 for k in ("ctc", "rnnt") if ok and isinstance(ok[0].get(k), dict)) or 1
        return (len(ok), decodings, p.stat().st_mtime)
    files = sorted(folder.glob("results_*.json"), key=rank)
    return files[-1] if files else None


def normalize(data: dict, slug: str, source: str, path: Path):
    """Turn the different result formats into rows:
    {model, variant, source, threads, load_sec, file -> {duration, sec, rtf, text}}"""
    variant = data.get("variant", "")
    base = dict(slug=slug, source=source, threads=data.get("threads"),
                load_sec=data.get("load_sec"), path=path.name)
    runs = {}  # decoding label -> {file: {...}}
    for r in data.get("results", []):
        if r.get("status") != "ok":
            continue
        dur = r.get("duration_sec")
        if "text" in r:  # single-decoding formats
            label = data.get("decoding") if isinstance(data.get("decoding"), str) else ""
            runs.setdefault(label, {})[r["file"]] = dict(duration=dur, sec=r.get("inference_sec"),
                                                        rtf=r.get("rtf"), text=r["text"])
        for dec in ("ctc", "rnnt"):
            if isinstance(r.get(dec), dict):  # local IndicConformer format
                d = r[dec]
                runs.setdefault(dec, {})[r["file"]] = dict(duration=dur, sec=d.get("inference_sec"),
                                                          rtf=d.get("rtf"), text=d.get("text", ""))
            elif f"{dec}_text" in r:  # Colab IndicConformer format
                runs.setdefault(dec, {})[r["file"]] = dict(duration=dur, sec=r.get(f"{dec}_sec"),
                                                          rtf=r.get(f"{dec}_rtf"), text=r[f"{dec}_text"])
    rows = []
    for dec, files in runs.items():
        label = " ".join(x for x in (variant, dec.upper() if dec in ("ctc", "rnnt") else "") if x)
        rows.append({**base, "variant": label, "files": files})
    return rows


def collect(include_colab: bool):
    rows = []
    for slug in MODEL_INFO:
        f = latest_json(ROOT / slug / "outputs")
        if f:
            rows += normalize(json.loads(f.read_text(encoding="utf-8")), slug, "local", f)
    if include_colab:
        for f in sorted((ROOT / "notebooks" / "outputs").glob("*_results_*.json")):
            slug = next((s for s in MODEL_INFO if f.name.startswith(s)), None)
            if slug:
                rows += normalize(json.loads(f.read_text(encoding="utf-8")), slug, "Colab", f)
    return rows


# --------------------------------------------------------------------------- #
def words(text: str):
    return re.findall(r"\S+", text or "")


def repeat_pct(text: str) -> float:
    w = words(text)
    if len(w) < 5:
        return 0.0
    _, top = Counter(w).most_common(1)[0]
    return 100.0 * (top - 1) / len(w)


def score(ref: str, hyp: str):
    try:
        import jiwer
    except ImportError:
        return None, None
    return jiwer.wer(ref, hyp), jiwer.cer(ref, hyp)


def fmt(x, spec):
    return format(x, spec) if isinstance(x, (int, float)) else "-"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Compare ASR model results")
    parser.add_argument("--no-colab", action="store_true", help="ignore notebooks/outputs/*.json")
    args = parser.parse_args()

    rows = collect(not args.no_colab)
    if not rows:
        print("No results found. Run test_model.py in the model folders first.")
        return 1

    audio_files = sorted({f for r in rows for f in r["files"]})
    refs = {}
    for a in audio_files:
        ref = (AUDIO_DIR / a).with_suffix(".txt")
        if ref.exists():
            refs[a] = ref.read_text(encoding="utf-8").strip()

    out = []
    out.append(f"# Bangla ASR model comparison\n\nGenerated {datetime.now():%Y-%m-%d %H:%M}. "
               f"Sources: latest `results_*.json` per model folder"
               f"{' + Colab JSONs in `notebooks/outputs/`' if not args.no_colab else ''}.\n")
    if not refs:
        out.append("> No reference transcripts found, so **WER/CER cannot be computed**. Add "
                   "`sample-audio/<audio name>.txt` with the correct text and run this script again.\n")

    # ---- summary table ----
    out.append("## Summary\n")
    out.append("| Model | Run | Source | Threads | Load (s) | Overall RTF | License | Params | Streaming | Fine-tune |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        info = MODEL_INFO[r["slug"]]
        ok = [v for v in r["files"].values() if v.get("sec") is not None and v.get("duration")]
        rtf = sum(v["sec"] for v in ok) / sum(v["duration"] for v in ok) if ok else None
        out.append(f"| {info['name']} | {r['variant'] or '-'} | {r['source']} | {r['threads'] or '-'} | "
                   f"{fmt(r['load_sec'], '.1f')} | {fmt(rtf, '.3f')} | {info['license']} | "
                   f"{info['params']} | {info['streaming']} | {info['finetune']} |")
    out.append("\nRTF = processing time / audio length (lower is faster; 0.1 = 10x faster than real time).  \n"
               "Load = one-time startup, including importing the library (NeMo alone takes ~20-45 s to "
               "import on Windows; the NeMo model weights themselves load in ~5-6 s). A server pays this "
               "once at start, not per request.\n")

    # ---- per file ----
    for a in audio_files:
        out.append(f"## {a}\n")
        if a in refs:
            out.append(f"**Reference:** {refs[a]}\n")
        out.append("| Model | Run | Source | RTF | Words | Repeat % | WER | CER |")
        out.append("|---|---|---|---|---|---|---|---|")
        for r in rows:
            v = r["files"].get(a)
            if not v:
                continue
            wer = cer = None
            if a in refs:
                wer, cer = score(refs[a], v["text"])
            rp = repeat_pct(v["text"])
            out.append(f"| {MODEL_INFO[r['slug']]['name']} | {r['variant'] or '-'} | {r['source']} | "
                       f"{fmt(v.get('rtf'), '.3f')} | {len(words(v['text']))} | "
                       f"{rp:.0f}%{' ⚠️' if rp > 20 else ''} | "
                       f"{fmt(wer * 100 if wer is not None else None, '.1f')} | "
                       f"{fmt(cer * 100 if cer is not None else None, '.1f')} |")
        out.append("\n**Transcripts**\n")
        for r in rows:
            v = r["files"].get(a)
            if v:
                label = f"{MODEL_INFO[r['slug']]['name']}" + (f" ({r['variant']})" if r["variant"] else "")
                out.append(f"- **{label}, {r['source']}:** {v['text'] or '<empty>'}")
        out.append("")

    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"comparison_{datetime.now():%Y%m%d_%H%M%S}.md"
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    print(f"\nSaved: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
