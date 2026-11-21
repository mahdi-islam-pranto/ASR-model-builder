"""
All settings for the fine-tuning scripts, in one place.

Per-machine changes (paths, batch size for a bigger GPU, ...) go in `config_local.py` next to this file,
so this file stays the same on every machine. Example config_local.py:

    BATCH_SIZE = 32          # 24 GB GPU
    GRAD_ACCUM = 1
    DATA_DIR = Path("/mnt/bigdisk/moonshine_data")

`config_local.py` is executed at the end of this file, so it can override any name below.
The `--quick` flag of every script calls apply_quick() (tiny end-to-end test run).
"""

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

# ---------------------------------------------------------------- run + paths
RUN_NAME = "run1_indicvoices_phone"      # change for every new experiment, so results stay separate
DATA_DIR = HERE / "data"                 # downloads (raw/) + processed datasets (processed/), ~10 GB for 60 h
RUNS_DIR = HERE / "runs"                 # one folder per RUN_NAME: checkpoints, logs, final model, results
MY_AUDIO_DIR = PROJECT_ROOT / "sample-audio"   # your own recordings (+ optional <name>.txt references)

BASE_MODEL = "Sanjidh090/moonshine-base-bn"    # or a local folder, e.g. PROJECT_ROOT / "moonshine_base_bn" / "model"

# ---------------------------------------------------------------- training data 1: IndicVoices Bengali
# Real extempore + conversational speech; two-person conversations were recorded over an 8 kHz phone line.
# CC-BY-4.0, gated: accept the conditions on https://huggingface.co/datasets/ai4bharat/IndicVoices first.
IV_REPO = "ai4bharat/IndicVoices"
IV_TRAIN_FILES = [f"bengali/train-{i:05d}-of-00096.parquet" for i in range(10)]   # 10 of 96 shards, ~410 MB each
IV_VALID_FILES = ["bengali/valid-00000-of-00001.parquet"]
IV_MAX_TRAIN_HOURS = 60       # stop collecting after this many hours
IV_TEXT_COL = None            # None = auto-detect. Run `prepare_data.py --inspect` and set it if needed
IV_KEEP = None                # e.g. {"task_name": ["conversation"]}: keep only some recording types

# ---------------------------------------------------------------- training data 2: clean rehearsal mix
# Bangladeshi read speech, mixed in so the model does not forget clean audio. CC-BY-4.0.
SUBAKKO_REPO = "SUST-CSE-Speech/SUBAK.KO"
SUBAKKO_FILES = [f"data/train-{i:05d}-of-00047.parquet" for i in range(4)]       # ~4 h per shard
CLEAN_MIX_RATIO = 0.20        # clean hours = 20% of the training set. 0 = off

# ---------------------------------------------------------------- test data (never trained on)
BANSPEECH_REPO = "SUST-CSE-Speech/banspeech"
BANSPEECH_SPONT = ["political_talkshow", "celebrity_interview", "drama_series"]
BANSPEECH_CLEAN = ["television_news", "audio_books"]
TEST_CLIPS_PER_SPLIT = 300
DEV_CLIPS = 300

# ---------------------------------------------------------------- filters for training clips
MIN_SEC, MAX_SEC = 1.0, 20.0
MAX_LABEL_TOKENS = 180        # the decoder has 194 positions
DROP_LATIN = True             # our rule: English words are written in Bangla script
DROP_DIGITS = True            # our rule: numbers are written as words

# ---------------------------------------------------------------- augmentation
PHONE_SIM_PROB = 0.5          # chance a wide-band training clip becomes phone audio (new random version each epoch)

# ---------------------------------------------------------------- training
LEARNING_RATE = 3e-5
NUM_EPOCHS = 3
MAX_STEPS = -1                # -1 = use NUM_EPOCHS; > 0 overrides
BATCH_SIZE = 16               # T4 16 GB: 16. 24 GB GPU: 32 (then GRAD_ACCUM = 1)
GRAD_ACCUM = 2                # effective batch = BATCH_SIZE * GRAD_ACCUM = 32
EVAL_BATCH_SIZE = 32
WARMUP = 0.10
WEIGHT_DECAY = 0.01
EVAL_STEPS = 250              # dev evaluation + checkpoint every N updates
SAVE_TOTAL_LIMIT = 2
PRECISION = "auto"            # auto (bf16 if the GPU supports it, else fp16) | fp16 | bf16 | fp32
NUM_WORKERS = 2               # dataloader worker processes (augmentation runs there)
SEED = 42

QUICK = False


def apply_quick():
    """Tiny end-to-end run (~10-15 min on a GPU) to check that everything works."""
    global QUICK, RUN_NAME, IV_TRAIN_FILES, IV_MAX_TRAIN_HOURS, SUBAKKO_FILES
    global TEST_CLIPS_PER_SPLIT, DEV_CLIPS, MAX_STEPS, EVAL_STEPS
    QUICK = True
    RUN_NAME = "quick_test"
    IV_TRAIN_FILES = IV_TRAIN_FILES[:1]
    IV_MAX_TRAIN_HOURS = min(IV_MAX_TRAIN_HOURS, 0.5)
    SUBAKKO_FILES = SUBAKKO_FILES[:1]
    TEST_CLIPS_PER_SPLIT = min(TEST_CLIPS_PER_SPLIT, 30)
    DEV_CLIPS = min(DEV_CLIPS, 40)
    MAX_STEPS = 60
    EVAL_STEPS = 30


def processed_dir() -> Path:
    """Prepared datasets. Shared by all runs that use the same data settings."""
    return DATA_DIR / ("processed_quick" if QUICK else "processed")


def raw_dir() -> Path:
    return DATA_DIR / "raw"


def run_dir() -> Path:
    return RUNS_DIR / RUN_NAME


def data_settings() -> dict:
    """Everything that changes the prepared data. If it changes, prepare_data.py rebuilds."""
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in dict(
        IV_REPO=IV_REPO, IV_TRAIN_FILES=IV_TRAIN_FILES, IV_VALID_FILES=IV_VALID_FILES,
        IV_MAX_TRAIN_HOURS=IV_MAX_TRAIN_HOURS, IV_TEXT_COL=IV_TEXT_COL, IV_KEEP=IV_KEEP,
        SUBAKKO_REPO=SUBAKKO_REPO, SUBAKKO_FILES=SUBAKKO_FILES, CLEAN_MIX_RATIO=CLEAN_MIX_RATIO,
        BANSPEECH_REPO=BANSPEECH_REPO, BANSPEECH_SPONT=BANSPEECH_SPONT, BANSPEECH_CLEAN=BANSPEECH_CLEAN,
        TEST_CLIPS_PER_SPLIT=TEST_CLIPS_PER_SPLIT, DEV_CLIPS=DEV_CLIPS, MIN_SEC=MIN_SEC, MAX_SEC=MAX_SEC,
        MAX_LABEL_TOKENS=MAX_LABEL_TOKENS, DROP_LATIN=DROP_LATIN, DROP_DIGITS=DROP_DIGITS,
        BASE_MODEL=BASE_MODEL, SEED=SEED,
    ).items()}


# ---------------------------------------------------------------- per-machine overrides
_local = HERE / "config_local.py"
if _local.exists():
    exec(compile(_local.read_text(encoding="utf-8"), str(_local), "exec"))
