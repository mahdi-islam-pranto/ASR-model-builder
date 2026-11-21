"""
STEP 4: score the fine-tuned model and compare with the baseline.

  python evaluate_model.py              # uses runs/<run>/final_model
  python evaluate_model.py --zip        # also make final_model.zip (to copy the model to another machine)

Same test sets and the same production decoding (beam 5) as baseline.py, so the numbers are comparable.

Output in runs/<run>/:
  final_scores.csv     baseline + fine-tuned rows
  results.md           before/after table, examples, your own recordings side by side

How to read the table (README section 3.4): the main score is test_bs_spont_phone (Bangladeshi spontaneous
speech through a simulated phone line). test_bs_clean must not get worse by more than ~1-2 WER points.
"""

import json
import shutil

import pandas as pd

import common as K
import config as C

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus", ".webm"}


def main(args):
    run = C.run_dir()
    final = run / "final_model"
    if not (final / "model.safetensors").exists():
        raise SystemExit(f"No fine-tuned model in {final}. Run train.py first.")
    base_csv = C.processed_dir() / "baseline_scores.csv"
    if not base_csv.exists():
        raise SystemExit("No baseline scores. Run baseline.py first.")

    sets = K.load_eval_sets()
    tokenizer = K.load_tokenizer()
    model = K.load_model(final)
    print(f"device: {K.device()} | model: {final}")
    ft_df, ft_hyps = K.evaluate_sets(model, tokenizer, sets, "finetuned")
    base_df = pd.read_csv(base_csv)
    base_hyps = json.loads((C.processed_dir() / "baseline_hyps.json").read_text(encoding="utf-8"))

    both = pd.concat([base_df, ft_df])
    both.to_csv(run / "final_scores.csv", index=False)
    cmp = both.pivot(index="test_set", columns="model", values=["WER", "CER"])
    table = pd.DataFrame({
        "WER before": cmp[("WER", "baseline")], "WER after": cmp[("WER", "finetuned")],
        "WER change": cmp[("WER", "finetuned")] - cmp[("WER", "baseline")],
        "CER before": cmp[("CER", "baseline")], "CER after": cmp[("CER", "finetuned")],
        "CER change": cmp[("CER", "finetuned")] - cmp[("CER", "baseline")],
    }).round(4)
    print("\n" + table.to_string())

    md = [f"# Results: `{C.RUN_NAME}`", "", "Production decoding (beam 5). Negative change = better.", "",
          table.reset_index().to_markdown(index=False) if _has_tabulate() else "```\n" + table.to_string() + "\n```", ""]
    info_file = final / "training_info.json"
    if info_file.exists():
        info = json.loads(info_file.read_text(encoding="utf-8"))
        md += [f"Training: {info.get('train_hours')} h, LR {info.get('learning_rate')}, epochs {info.get('epochs')}, "
               f"{info.get('training_minutes')} min on {info.get('gpu')}, best dev WER (greedy) "
               f"{info.get('best_dev_wer_greedy')}.", ""]
        info["scores"] = both.to_dict("records")
        info_file.write_text(json.dumps(info, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    name = "test_bs_spont_phone"
    if name in sets and name in base_hyps:
        md += [f"## Examples: {name}", ""]
        for (_, ref), b, f in list(zip(sets[name], base_hyps[name], ft_hyps[name]))[:10]:
            md += [f"- **ref:** {ref}", f"  - before: {b}", f"  - after: {f}"]
        md.append("")

    # ---- own recordings, old vs new model
    files = sorted(p for p in C.MY_AUDIO_DIR.iterdir() if p.suffix.lower() in AUDIO_EXTS) \
        if C.MY_AUDIO_DIR.is_dir() else []
    if files:
        print(f"\nOwn recordings in {C.MY_AUDIO_DIR}:")
        base_model = K.load_model()
        md += ["## Own recordings", ""]
        for p in files:
            b = K.transcribe_file(base_model, tokenizer, p)
            f = K.transcribe_file(model, tokenizer, p)
            md += [f"### {p.name}", "", f"- **before:** {b}", f"- **after:** {f}"]
            print(f"\n== {p.name}\n  before: {b}\n  after : {f}")
            ref = p.with_suffix(".txt")
            if ref.exists():
                r = K.normalize_text(ref.read_text(encoding="utf-8"))
                (wb, cb), (wf, cf) = K.score([r], [b]), K.score([r], [f])
                md.append(f"- WER {wb:.3f} → {wf:.3f}, CER {cb:.3f} → {cf:.3f}")
                print(f"  WER {wb:.3f} -> {wf:.3f} | CER {cb:.3f} -> {cf:.3f}")
            md.append("")
    else:
        print(f"(no audio in {C.MY_AUDIO_DIR}; skipping own recordings)")

    (run / "results.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nreport: {run / 'results.md'}")
    if args.zip:
        z = shutil.make_archive(str(run / "final_model"), "zip", str(final))
        print(f"zip: {z}")


def _has_tabulate():
    try:
        import tabulate  # noqa: F401  (pandas needs it for to_markdown)
        return True
    except ImportError:
        return False


def parser():
    p = K.base_parser("Step 4: evaluate the fine-tuned model vs the baseline")
    p.add_argument("--zip", action="store_true", help="also create runs/<run>/final_model.zip")
    return p


if __name__ == "__main__":
    a = parser().parse_args()
    K.setup(a, "evaluate")
    main(a)
