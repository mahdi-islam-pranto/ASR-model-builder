"""
Run all steps in order: prepare_data -> baseline -> train -> evaluate.

  python run_all.py --quick       # tiny end-to-end check first (~10-15 min on a GPU)
  python run_all.py               # the real run

Every step skips work that is already done (prepared data, baseline) and training resumes from the last
checkpoint, so after a crash or disconnect you just run the same command again.

On a remote Linux machine, run it so it survives closing the terminal:
  nohup python run_all.py > run_all.out 2>&1 &      (watch with: tail -f run_all.out)
or inside `tmux` / `screen`.
"""

import baseline
import common as K
import evaluate_model
import prepare_data
import train


def main():
    p = K.base_parser("Run all fine-tuning steps")
    p.add_argument("--force-data", action="store_true", help="rebuild the prepared data")
    p.add_argument("--fresh", action="store_true", help="restart training from scratch (delete checkpoints)")
    p.add_argument("--zip", action="store_true", help="zip the final model at the end")
    a = p.parse_args()
    K.setup(a, "run_all")

    a.force, a.inspect = a.force_data, False
    print("\n##### STEP 1/4: prepare data #####")
    prepare_data.main(a)
    a.force = False
    print("\n##### STEP 2/4: baseline #####")
    baseline.main(a)
    print("\n##### STEP 3/4: train #####")
    train.main(a)
    print("\n##### STEP 4/4: evaluate #####")
    evaluate_model.main(a)
    print("\nAll done.")


if __name__ == "__main__":
    main()
