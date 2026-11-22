"""
STEP 2: score the ORIGINAL model on all test sets, before any training.

  python baseline.py            # skips if the baseline already exists for this data
  python baseline.py --force

Without a baseline you cannot tell whether training helped. Decoding = production settings
(beam 5, no_repeat_ngram_size 3, repetition_penalty 1.2), exactly like moonshine_base_bn/moonshine_bn.py.

Output (shared by all runs on the same prepared data):
  data/processed/baseline_scores.csv
  data/processed/baseline_hyps.json     (transcripts, used for side-by-side comparison later)
"""

import json

import common as K
import config as C


def main(args):
    out = C.processed_dir()
    csv, hyp_file = out / "baseline_scores.csv", out / "baseline_hyps.json"
    if csv.exists() and hyp_file.exists() and not args.force:
        print(f"Baseline already exists: {csv} (use --force to recompute)")
        print(open(csv, encoding="utf-8").read())
        return

    sets = K.load_eval_sets()
    tokenizer = K.load_tokenizer()
    model = K.load_model()
    print(f"device: {K.device()} | precision for decoding: {K.precision()}")
    df, hyps = K.evaluate_sets(model, tokenizer, sets, "baseline")
    df.to_csv(csv, index=False)
    hyp_file.write_text(json.dumps(hyps, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n" + df.to_string(index=False))
    own = K.own_test_files()
    if own:
        print(f"\nYour own calls ({len(own)} labelled files in {C.OWN_TEST_DIR}), for information "
              "(evaluate_model.py scores them again with both models):")
        K.evaluate_own(model, tokenizer, "baseline", own)
    else:
        print(f"\nNo labelled calls in {C.OWN_TEST_DIR} yet (x.wav + x.txt). See README section 3.0.")

    name = "test_bs_spont_phone"
    if name in sets:
        print(f"\nExamples from {name} (what the phone problem looks like):")
        for (_, ref), hyp in list(zip(sets[name], hyps[name]))[:5]:
            print("  REF:", ref)
            print("  HYP:", hyp, "\n")
    print(f"saved: {csv}")


def parser():
    p = K.base_parser("Step 2: baseline scores of the original model")
    p.add_argument("--force", action="store_true", help="recompute even if it exists")
    return p


if __name__ == "__main__":
    a = parser().parse_args()
    K.setup(a, "baseline")
    main(a)
