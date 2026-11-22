"""
All settings for the fine-tuning scripts, in one place.

Per-machine changes (paths, batch size for a bigger GPU, ...) go in `config_local.py` next to this file,
so this file stays the same on every machine. Example config_local.py:

    BATCH_SIZE = 32          # 24 GB GPU
    GRAD_ACCUM = 1
    DATA_DIR = Path("/mnt/bigdisk/moonshine_data")

`config_local.py` is executed at the end of this file, so it can override any name below.
The Kaggle / Colab notebooks write their own config_local.py (paths on Google Drive / Kaggle disks).
The `--quick` flag of every script calls apply_quick() (tiny end-to-end test run).
"""

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

# ---------------------------------------------------------------- run + paths
RUN_NAME = "stage1_subakko_phone"        # change for every new experiment, so results stay separate
DATA_DIR = HERE / "data"                 # processed datasets (processed/), ~7 GB for 60 h
RAW_DIR = None                           # downloaded parquet files. None = DATA_DIR/raw (~11 GB for 60 h)
DELETE_RAW_AFTER_PREP = False            # True = delete downloads after prepare_data (saves disk on Kaggle/Colab)
RUNS_DIR = HERE / "runs"                 # one folder per RUN_NAME: checkpoints, logs, final model, results
MY_AUDIO_DIR = PROJECT_ROOT / "sample-audio"   # your recordings: transcribed before/after (a .txt next to it = scored)
OWN_TEST_DIR = HERE / "own_test_set"     # YOUR labelled call clips: x.wav + x.txt pairs. Scored in baseline + evaluate

BASE_MODEL = "Sanjidh090/moonshine-base-bn"    # or a local folder / a previous final_model (stage 2 starts from stage 1)

# ---------------------------------------------------------------- training data
# A list of sources. Each one: Hugging Face dataset repo, which parquet files, column names, how many hours.
#   phone_sim=True : the clip becomes phone audio with probability PHONE_SIM_PROB (new random version every epoch),
#                    otherwise only the volume changes. Real 8 kHz audio is never phone-simulated again.
#   keep           : optional filter, e.g. {"dialect": ["Sylhet", "Chittagong"]}
# Hours are spread evenly over the files, so more files = more different speakers/books.
SUBAKKO_REPO = "SUST-CSE-Speech/SUBAK.KO"     # Bangladeshi standard Bangla, human-checked, CC-BY-4.0


def subakko_train_files(n):
    """n of the 47 SUBAK.KO train files (~470 MB, ~4.3 h each), spread over the whole set."""
    return [f"data/train-{round(i * 47 / n):05d}-of-00047.parquet" for i in range(n)]


# Stage 1: standard Bangladeshi Bangla. ~30% of the clips are dropped (digits / English letters), so one file
# gives ~3 h: 20 files ~ 60 h (~9.5 GB download). In config_local.py you can change it,
# e.g. TRAIN_SOURCES[0].update(files=subakko_train_files(10), hours=30) for a shorter run.
TRAIN_SOURCES = [
    dict(name="subakko", repo=SUBAKKO_REPO, files=subakko_train_files(20),
         text_col="transcription", hours=60, phone_sim=True),
]

# Stage 2 (later, NOT now): regional dialects. Machine-made labels (Whisper) in dialect spelling, CC-BY-NC.
# Start from the stage 1 model (BASE_MODEL = runs/stage1_subakko_phone/final_model) and keep ~30% SUBAK.KO:
# TRAIN_SOURCES = [
#     dict(name="dialect", repo="niloycste68/Bangali_local_dialect_ASR_HF_Dataset",
#          files=[f"data/train-{i:05d}-of-00027.parquet" for i in range(0, 27, 2)],
#          text_col="sentence", hours=40, phone_sim=True, keep=None, category_col="dialect"),
#     dict(name="subakko", repo=SUBAKKO_REPO,
#          files=[f"data/train-{i:05d}-of-00047.parquet" for i in range(20, 47, 6)],   # other files than stage 1
#          text_col="transcription", hours=17, phone_sim=True),
# ]

# ---------------------------------------------------------------- dev + test data (never trained on)
# dev  = SUBAK.KO validation clips, half clean + half phone-simulated. Picks the best checkpoint.
# tests: SUBAK.KO test (clean + the same clips phone-simulated) and BanSpeech domains (Bangladeshi broadcast)
SUBAKKO_VALID_FILES = ["data/validation-00000-of-00005.parquet"]
SUBAKKO_TEST_FILES = ["data/test-00000-of-00005.parquet", "data/test-00003-of-00005.parquet"]
BANSPEECH_REPO = "SUST-CSE-Speech/banspeech"
BANSPEECH_SPONT = ["political_talkshow", "celebrity_interview", "drama_series"]   # free conversation
BANSPEECH_MEDICINE = ["medicine"]                                                 # medical words
BANSPEECH_CLEAN = ["television_news", "audio_books"]
TEST_CLIPS_PER_SPLIT = 300
DEV_CLIPS = 300

# ---------------------------------------------------------------- filters for training clips
MIN_SEC, MAX_SEC = 1.0, 20.0
MAX_LABEL_TOKENS = 180        # the decoder has 194 positions
DROP_LATIN = True             # our rule: English words are written in Bangla script
DROP_DIGITS = True            # our rule: numbers are written as words

# ---------------------------------------------------------------- augmentation
PHONE_SIM_PROB = 0.6          # 60% phone audio (the target), 40% clean (so clean audio is not forgotten)

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
    global QUICK, RUN_NAME, TRAIN_SOURCES, SUBAKKO_TEST_FILES, TEST_CLIPS_PER_SPLIT, DEV_CLIPS, MAX_STEPS, EVAL_STEPS
    QUICK = True
    RUN_NAME = "quick_test"
    TRAIN_SOURCES = [dict(s, files=s["files"][:1], hours=min(s["hours"], 0.5)) for s in TRAIN_SOURCES]
    SUBAKKO_TEST_FILES = SUBAKKO_TEST_FILES[:1]
    TEST_CLIPS_PER_SPLIT = min(TEST_CLIPS_PER_SPLIT, 30)
    DEV_CLIPS = min(DEV_CLIPS, 40)
    MAX_STEPS = 60
    EVAL_STEPS = 30


def processed_dir() -> Path:
    """Prepared datasets. Shared by all runs that use the same data settings."""
    return Path(DATA_DIR) / ("processed_quick" if QUICK else "processed")


def raw_dir() -> Path:
    return Path(RAW_DIR) if RAW_DIR else Path(DATA_DIR) / "raw"


def run_dir() -> Path:
    return Path(RUNS_DIR) / RUN_NAME


def data_settings() -> dict:
    """Everything that changes the prepared data. If it changes, prepare_data.py rebuilds."""
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in dict(
        TRAIN_SOURCES=TRAIN_SOURCES, SUBAKKO_REPO=SUBAKKO_REPO, SUBAKKO_VALID_FILES=SUBAKKO_VALID_FILES,
        SUBAKKO_TEST_FILES=SUBAKKO_TEST_FILES, BANSPEECH_REPO=BANSPEECH_REPO, BANSPEECH_SPONT=BANSPEECH_SPONT,
        BANSPEECH_MEDICINE=BANSPEECH_MEDICINE, BANSPEECH_CLEAN=BANSPEECH_CLEAN,
        TEST_CLIPS_PER_SPLIT=TEST_CLIPS_PER_SPLIT, DEV_CLIPS=DEV_CLIPS, MIN_SEC=MIN_SEC, MAX_SEC=MAX_SEC,
        MAX_LABEL_TOKENS=MAX_LABEL_TOKENS, DROP_LATIN=DROP_LATIN, DROP_DIGITS=DROP_DIGITS,
        BASE_MODEL=str(BASE_MODEL), SEED=SEED,
    ).items()}


# ---------------------------------------------------------------- per-machine overrides
_local = HERE / "config_local.py"
if _local.exists():
    exec(compile(_local.read_text(encoding="utf-8"), str(_local), "exec"))
