"""
STEP 3: fine-tune.

  python train.py                 # starts, or RESUMES automatically from the last checkpoint of this run
  python train.py --fresh         # ignore existing checkpoints and start over

What happens (details in README section 2):
  * every step: BATCH_SIZE clips -> collator (phone simulation, padding, attention mask, labels with -100)
    -> model(audio, labels) -> cross-entropy loss -> backprop -> AdamW update
  * gradients of GRAD_ACCUM batches are added before one update (effective batch = BATCH_SIZE * GRAD_ACCUM)
  * learning rate: linear warm-up over the first 10% of updates, then linear decay to 0
  * every EVAL_STEPS updates: generate text for the dev set (greedy), compute WER/CER, save a checkpoint
  * at the end: reload the checkpoint with the LOWEST dev WER (not necessarily the last) and save it as
    runs/<run>/final_model  (+ fixed generation_config.json, tokenizer, training_info.json)

If the process is stopped (disconnect, crash, Ctrl+C), just run it again: it continues from the last
checkpoint (saved every EVAL_STEPS updates, so at most EVAL_STEPS updates are lost).
"""

import json
import math
import shutil
import time

import numpy as np

import common as K
import config as C


def plot_curves(hist, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
    if "loss" in hist:
        tr = hist.dropna(subset=["loss"])
        ax[0].plot(tr["step"], tr["loss"], label="train loss")
    if "eval_loss" in hist:
        ev = hist.dropna(subset=["eval_loss"])
        ax[0].plot(ev["step"], ev["eval_loss"], "o-", label="dev loss")
        ax[1].plot(ev["step"], ev["eval_wer"], "o-", label="dev WER")
        ax[1].plot(ev["step"], ev["eval_cer"], "o-", label="dev CER")
    for a in ax:
        a.set_xlabel("update step")
        a.legend()
        a.grid(alpha=.3)
    ax[0].set_title("loss")
    ax[1].set_title("dev error rate (greedy decoding)")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


def main(args):
    import pandas as pd
    import torch
    import transformers
    from datasets import load_from_disk
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback
    from transformers.trainer_utils import get_last_checkpoint

    data = C.processed_dir()
    if not (data / "prep_settings.json").exists():
        raise SystemExit(f"No prepared data in {data}. Run prepare_data.py first.")
    run = C.run_dir()
    ckpt_dir = run / "checkpoints"
    if args.fresh and ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)

    # numpy format: audio comes out as arrays (fast) instead of Python lists
    train_ds = load_from_disk(str(data / "train")).with_format("numpy")
    dev_ds = load_from_disk(str(data / "dev")).with_format("numpy")
    tokenizer = K.load_tokenizer()
    model = K.load_model()
    prec = K.precision()

    steps_per_epoch = math.ceil(len(train_ds) / (C.BATCH_SIZE * C.GRAD_ACCUM))
    total = C.MAX_STEPS if C.MAX_STEPS > 0 else steps_per_epoch * C.NUM_EPOCHS
    gpu = torch.cuda.get_device_name(0) if K.device() == "cuda" else "none (CPU, very slow)"
    print(f"GPU: {gpu} | precision: {prec} | transformers {transformers.__version__}")
    print(f"train: {len(train_ds)} clips ({np.sum(train_ds['length']) / K.SR / 3600:.1f} h) | dev: {len(dev_ds)} clips")
    print(f"batch {C.BATCH_SIZE} x accum {C.GRAD_ACCUM} = {C.BATCH_SIZE * C.GRAD_ACCUM} clips per update | "
          f"{steps_per_epoch} updates/epoch | {total} updates in total | LR {C.LEARNING_RATE}")

    def compute_metrics(pred):
        p = np.where(pred.predictions < 0, K.PAD_ID, pred.predictions)
        l = np.where(pred.label_ids < 0, K.PAD_ID, pred.label_ids)
        hyps = [K.normalize_text(s) for s in tokenizer.batch_decode(p, skip_special_tokens=True)]
        refs = [K.normalize_text(s) for s in tokenizer.batch_decode(l, skip_special_tokens=True)]
        w, c = K.score(refs, hyps)
        return {"wer": w, "cer": c}

    class SaveHistory(TrainerCallback):
        """Write log_history.csv + curves.png after every evaluation, so you can watch progress."""

        def on_evaluate(self, a, state, control, **kw):
            hist = pd.DataFrame(state.log_history)
            hist.to_csv(run / "log_history.csv", index=False)
            try:
                plot_curves(hist, run / "curves.png")
            except Exception as e:
                print("plot failed:", e)

    targs = Seq2SeqTrainingArguments(
        output_dir=str(ckpt_dir),
        per_device_train_batch_size=C.BATCH_SIZE,
        per_device_eval_batch_size=C.EVAL_BATCH_SIZE,
        gradient_accumulation_steps=C.GRAD_ACCUM,
        learning_rate=C.LEARNING_RATE,
        lr_scheduler_type="linear",
        warmup_steps=C.WARMUP,
        weight_decay=C.WEIGHT_DECAY,
        max_grad_norm=1.0,
        num_train_epochs=C.NUM_EPOCHS,
        max_steps=C.MAX_STEPS,
        fp16=(prec == "fp16"),
        bf16=(prec == "bf16"),
        eval_strategy="steps", eval_steps=C.EVAL_STEPS,
        save_strategy="steps", save_steps=C.EVAL_STEPS, save_total_limit=C.SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True, metric_for_best_model="wer", greater_is_better=False,
        predict_with_generate=True, generation_max_length=160, generation_num_beams=1,
        logging_steps=25, logging_first_step=True, report_to="none",
        remove_unused_columns=False,
        train_sampling_strategy="group_by_length", length_column_name="length",
        dataloader_num_workers=C.NUM_WORKERS,
        disable_tqdm=False,
        seed=C.SEED,
    )
    trainer = Seq2SeqTrainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=dev_ds,
                             data_collator=K.SpeechCollator(C.PHONE_SIM_PROB, seed=C.SEED),
                             compute_metrics=compute_metrics, processing_class=tokenizer,
                             callbacks=[SaveHistory()])

    last = get_last_checkpoint(str(ckpt_dir)) if ckpt_dir.exists() else None
    print("resuming from:", last) if last else print("starting fresh")
    t0 = time.time()
    trainer.train(resume_from_checkpoint=last)
    minutes = (time.time() - t0) / 60
    print(f"training time: {minutes:.1f} min")
    print(f"best checkpoint: {trainer.state.best_model_checkpoint} | best dev WER (greedy): {trainer.state.best_metric}")

    hist = pd.DataFrame(trainer.state.log_history)
    hist.to_csv(run / "log_history.csv", index=False)
    plot_curves(hist, run / "curves.png")

    final = run / "final_model"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))
    K.fixed_generation_config().save_pretrained(str(final))
    (final / "training_args.bin").unlink(missing_ok=True)
    report = json.loads((data / "data_report.json").read_text(encoding="utf-8"))
    info = {
        "base_model": str(C.BASE_MODEL), "run_name": C.RUN_NAME, "quick_test": C.QUICK,
        "train_clips": report.get("train_clips"), "train_hours": report.get("train_hours"),
        "train_by_source": report.get("train"), "data_settings": C.data_settings(),
        "phone_sim_prob": C.PHONE_SIM_PROB, "learning_rate": C.LEARNING_RATE, "epochs": C.NUM_EPOCHS,
        "max_steps": C.MAX_STEPS, "batch": C.BATCH_SIZE, "grad_accum": C.GRAD_ACCUM, "warmup": C.WARMUP,
        "precision": prec, "gpu": gpu, "training_minutes": round(minutes, 1),
        "best_checkpoint": trainer.state.best_model_checkpoint, "best_dev_wer_greedy": trainer.state.best_metric,
        "versions": {"transformers": transformers.__version__, "torch": torch.__version__},
    }
    (final / "training_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2, default=str),
                                              encoding="utf-8")
    print(f"final model saved: {final}")


def parser():
    p = K.base_parser("Step 3: fine-tune (resumes automatically)")
    p.add_argument("--fresh", action="store_true", help="delete this run's checkpoints and start over")
    return p


if __name__ == "__main__":
    a = parser().parse_args()
    K.setup(a, "train")
    main(a)
