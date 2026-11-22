"""
Builds moonshine_finetune_kaggle_colab.ipynb from the .py files in this folder.

The notebook carries a copy of every script (one %%writefile cell each), so you only upload ONE file to Kaggle
or Colab. After you change a .py file, run this again so the notebook gets the same code:

    python make_notebook.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "moonshine_finetune_kaggle_colab.ipynb"
CODE_FILES = ["config.py", "common.py", "prepare_data.py", "baseline.py", "train.py", "evaluate_model.py", "run_all.py"]

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text, hidden=False):
    meta = {"cellView": "form", "jupyter": {"source_hidden": True}} if hidden else {}
    cells.append({"cell_type": "code", "metadata": meta, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


md("""
# Fine-tune Moonshine-Base-BN: stage 1 (standard Bangladeshi Bangla + phone audio)

**Same pipeline as the .py scripts** in `Fine-tune-moonshine/` (this notebook writes them to disk and runs them),
so the results are identical on Kaggle, Colab, or your friend's GPU PC.

**What stage 1 does**
- trains on **SUBAK.KO** (human-checked standard Bangladeshi Bangla, ~60 h); 60% of the clips are turned into
  **8 kHz phone audio** on the fly, so the model learns the sound of your calls while the text stays correct
- picks the best checkpoint on a SUBAK.KO dev set (half clean, half phone)
- compares the original vs fine-tuned model on: SUBAK.KO test (clean + phone), BanSpeech talk-show/interview/drama
  (phone + clean), BanSpeech **medicine** (phone), news/audiobooks (clean), and **your own labelled calls**

**Which platform?**

| | Kaggle (recommended) | Colab free |
|---|---|---|
| GPU time | ~30 h/week, 12 h per session | a few h/day, varies |
| Browser closed / disconnect | "Save Version → Save & Run All" runs in the background | session dies; re-run all cells, training resumes from Drive |
| Setup | Settings: Accelerator **GPU T4 x2**, Internet **On** (needs a phone-verified account) | Runtime → Change runtime type → **T4 GPU** |

**How to use it:** run the cells from top to bottom. First with `QUICK = True` (~15 min, checks everything works),
then set `QUICK = False` and run everything again for the real training.
""")

md("""
## 1. Settings: the only cell you normally change
""")
code('''
QUICK = True                  # True = tiny test run first. Then False = the real run
RUN_NAME = "stage1_subakko_phone"
TRAIN_HOURS = 60              # SUBAK.KO hours (~3 usable h per downloaded file of ~470 MB)
PHONE_SIM_PROB = 0.6          # share of training clips turned into phone audio
NUM_EPOCHS = 3
LEARNING_RATE = 3e-5
BATCH_SIZE, GRAD_ACCUM = 16, 2   # T4: 16 x 2 = 32 clips per update. Out of memory -> 8, 4

# Colab only: keep the prepared data on Google Drive too (~7 GB for 60 h).
# False = rebuilt after every disconnect (~20-40 min); checkpoints are ALWAYS saved to Drive.
DATA_ON_DRIVE = False
''')

md("""
## 2. Platform, folders, GPU

- **Colab:** checkpoints, logs and the final model go to Google Drive (`MyDrive/moonshine_ft/runs`), so a
  disconnect loses at most 250 training updates. Downloads stay on the fast local disk.
- **Kaggle:** results go to `/kaggle/working/runs` (kept as the notebook's *Output* when you "Save Version").
  Downloads and prepared data go to `/tmp` (more space, not saved). To **continue** a run in a new session, add
  the previous version's output as an input (*Add Input → Your Work → this notebook*); the cell below copies the
  checkpoints back and training resumes.
- **Your own labelled calls** (the most important test): a folder of short clips, `x.wav` + `x.txt` (correct
  transcript). Colab: upload to `MyDrive/moonshine_ft/own_test_set/`. Kaggle: upload as a **private** dataset
  and add it as input; any input folder named `own_test_set` (or containing `.txt` + audio pairs) is found.
  These are patient calls: keep them private.
""")
code('''
import os, shutil, subprocess, sys, glob
from pathlib import Path

ON_KAGGLE = Path("/kaggle/working").exists()
ON_COLAB = "google.colab" in sys.modules or Path("/content").exists() and not ON_KAGGLE
print("platform:", "Kaggle" if ON_KAGGLE else "Colab" if ON_COLAB else "local")

if ON_KAGGLE:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"          # use one of the two T4s (the scripts are tested on 1 GPU)
    CODE = Path("/kaggle/working/Fine-tune-moonshine")
    DATA = Path("/tmp/moonshine_ft_data")
    RUNS = Path("/kaggle/working/runs")
    OWN = None
    for d in glob.glob("/kaggle/input/**/own_test_set", recursive=True):
        OWN = Path(d)
    if OWN is None:                                   # any input folder with audio + .txt pairs
        for txt in glob.glob("/kaggle/input/**/*.txt", recursive=True):
            if any(Path(txt).with_suffix(e).exists() for e in (".wav", ".mp3", ".flac", ".m4a", ".ogg")):
                OWN = Path(txt).parent
                break
    # continue a run from a previous version's output
    for old in glob.glob(f"/kaggle/input/**/runs/{RUN_NAME}", recursive=True):
        if not (RUNS / RUN_NAME / "checkpoints").exists() and Path(old, "checkpoints").exists():
            print("copying previous checkpoints from", old)
            shutil.copytree(old, RUNS / RUN_NAME, dirs_exist_ok=True)
elif ON_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    PERSIST = Path("/content/drive/MyDrive/moonshine_ft")
    CODE = Path("/content/Fine-tune-moonshine")
    DATA = PERSIST / "data" if DATA_ON_DRIVE else Path("/content/ft_data")
    RUNS = PERSIST / "runs"
    OWN = PERSIST / "own_test_set"
else:
    CODE = Path.cwd() / "Fine-tune-moonshine"
    DATA, RUNS, OWN = CODE / "data", CODE / "runs", CODE / "own_test_set"

CODE.mkdir(parents=True, exist_ok=True)
os.chdir(CODE)
print("code:", CODE, "| data:", DATA, "| runs:", RUNS, "| own test set:", OWN)
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"],
                     capture_output=True, text=True).stdout or "NO GPU! Turn the GPU on in the settings.")
print(subprocess.run(["df", "-h", "/tmp" if ON_KAGGLE else "/content" if ON_COLAB else "."],
                     capture_output=True, text=True).stdout)
''')

md("""
## 3. Install packages
Colab and Kaggle already have PyTorch with CUDA. We add the exact `transformers` version the scripts were
tested with (Moonshine needs transformers 5.x). Takes ~1-2 min. If pip prints dependency *warnings* about other
preinstalled packages, that is fine.
""")
code('''
!pip -q install "transformers==5.22.2" "accelerate>=1.10" "datasets>=4.0" huggingface_hub jiwer soundfile "librosa>=0.10" tabulate
import torch, transformers, datasets
print("torch", torch.__version__, "| transformers", transformers.__version__, "| datasets", datasets.__version__,
      "| cuda", torch.cuda.is_available())
''')

md("""
## 4. Write the code files
These cells are copies of the `.py` files (generated by `make_notebook.py`). You do not need to read or change
them here: all settings come from cell 1, which is written to `config_local.py` below. If you want to understand
a step, the docstring at the top of each file explains it (and README.md section 2).
""")
for f in CODE_FILES:
    code(f"%%writefile {f}\n" + (HERE / f).read_text(encoding="utf-8"), hidden=True)

code('''
# per-platform settings -> config_local.py (overrides config.py; same mechanism as on a local GPU PC)
local = f"""
DATA_DIR = Path({str(DATA)!r})
RAW_DIR = Path({str(Path('/tmp/moonshine_ft_raw') if ON_KAGGLE else Path('/content/ft_raw') if ON_COLAB else DATA / 'raw')!r})
DELETE_RAW_AFTER_PREP = {bool(ON_KAGGLE or ON_COLAB)}
RUNS_DIR = Path({str(RUNS)!r})
OWN_TEST_DIR = Path({str(OWN)!r}) if {OWN is not None} else HERE / "own_test_set"
MY_AUDIO_DIR = HERE / "sample-audio"
RUN_NAME = {RUN_NAME!r}
TRAIN_SOURCES[0].update(files=subakko_train_files(max(1, min(47, round({TRAIN_HOURS} / 3)))), hours={TRAIN_HOURS})
PHONE_SIM_PROB = {PHONE_SIM_PROB}
NUM_EPOCHS = {NUM_EPOCHS}
LEARNING_RATE = {LEARNING_RATE}
BATCH_SIZE, GRAD_ACCUM = {BATCH_SIZE}, {GRAD_ACCUM}
"""
Path("config_local.py").write_text(local, encoding="utf-8")
print(local)
Q = "--quick" if QUICK else ""
print("mode:", "QUICK test" if QUICK else "REAL run")
''')

md("""
## 5. Step 1: prepare the data
Downloads SUBAK.KO train files, the SUBAK.KO validation/test files and BanSpeech, cleans the text, drops clips
that do not fit our rules (digits, English letters, too long...), and builds:
`train/`, `dev/`, `tests/*`. It prints example texts (raw → cleaned label), sample rates and how many clips were
dropped and why. Real run: ~10 GB download, ~20-40 min. Skipped automatically if already built.
""")
code('''
!python prepare_data.py $Q
''')
code('''
# listen: a clean clip and the same clip through the simulated phone line
from IPython.display import Audio, Image, display
import config as C
ex = Path(C.DATA_DIR) / ("processed_quick" if QUICK else "processed") / "examples"
for w in sorted(ex.glob("*.wav"))[:4]:
    print(w.name); display(Audio(str(w)))
if (ex / "phone_sim_spectrogram.png").exists():
    display(Image(str(ex / "phone_sim_spectrogram.png")))
''')

md("""
## 6. Step 2: baseline (the original model, before training)
Without this you cannot tell whether training helped. Uses the production decoding (beam 5). Lower WER/CER =
better. Also scores your own labelled calls if they were found.
""")
code('''
!python baseline.py $Q
''')

md("""
## 7. Step 3: train
Every 250 updates: dev WER/CER + a checkpoint. If the session stops, run all cells again: it **resumes** from
the last checkpoint. Watch `eval_wer` going down. Train loss going down while dev WER goes up = overfitting
(the best checkpoint is kept anyway).
""")
code('''
!python train.py $Q
''')
code('''
import config as C
run = Path(C.RUNS_DIR) / ("quick_test" if QUICK else RUN_NAME)
if (run / "curves.png").exists():
    display(Image(str(run / "curves.png")))
''')

md("""
## 8. Step 4: evaluate and compare
Same test sets + same decoding as the baseline. Read the table like this:
- **own_test_set**: the real answer (your calls). Lower after = success.
- `*_phone` rows should go down (that is what stage 1 trains for).
- `test_subakko_clean` / `test_bs_clean` must not get worse by more than ~1-2 WER points.
""")
code('''
!python evaluate_model.py $Q --zip
''')
code('''
from IPython.display import Markdown
res = run / "results.md"
if res.exists():
    display(Markdown(res.read_text(encoding="utf-8")))
''')

md("""
## 9. Download the model
`final_model.zip` (~230 MB) is in the run folder:
- **Colab:** it is on Google Drive (`MyDrive/moonshine_ft/runs/<run>/`); or run the cell below.
- **Kaggle:** after "Save Version" it is in the notebook's *Output* tab (`runs/<run>/final_model.zip`).

On your PC unzip it to `Fine-tune-moonshine/models/<run>/` and compare with
`python Fine-tune-moonshine/test_finetuned.py --model Fine-tune-moonshine/models/<run>` (README section 3.6).
""")
code('''
z = run / "final_model.zip"
print(z, z.exists())
if ON_COLAB and z.exists():
    from google.colab import files
    files.download(str(z))
''')

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(cells):
    c["id"] = f"c{i:02d}"
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"written {OUT} ({len(cells)} cells)")
