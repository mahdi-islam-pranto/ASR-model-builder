# Fine-tune Moonshine-Base-BN on open Bangla conversational speech (experiment 1)

**Goal:** before we label our own call recordings, find out whether fine-tuning `Sanjidh090/moonshine-base-bn`
on **open** conversational Bangla speech, plus **simulated phone-line audio**, makes it better on
call-center-like audio. Along the way, learn the full fine-tuning and testing process on a free Colab T4.

This is a **trial run**, not the final model. The final model needs our own labelled calls
([../MOONSHINE_FINETUNE_PLAN.md](../MOONSHINE_FINETUNE_PLAN.md)). This experiment answers two questions first:

1. Does the training pipeline work end to end (data → training → evaluation → model on the PC)?
2. How much does phone + conversational training help, and does it hurt clean audio?

| File | What it is |
|---|---|
| [moonshine_finetune_colab.ipynb](moonshine_finetune_colab.ipynb) | Colab notebook: data, baseline, training, evaluation, saving. Every step is explained in the notebook |
| [test_finetuned.py](test_finetuned.py) | Run on your PC: original vs fine-tuned model on `sample-audio/`, side-by-side report |
| `models/` | Put downloaded fine-tuned models here (git-ignored, ~245 MB each) |
| `outputs/` | Reports written by `test_finetuned.py` |

---

## 1. Which dataset? (survey and recommendation)

### 1.1 What the model was already trained on

The model card says it was trained on **Lipi-Ghor-bn-882-SSTT**: ~880 h of Bangla from 1,019 **YouTube videos**
with caption-based transcripts (CC-BY-4.0, gated, 50 GB). It is wide-band, mostly clean, multi-domain audio.
That is why the model is good on news, movie and AI voice audio, and weak on 8 kHz phone calls: **it has
never heard phone audio.**

### 1.2 The collection you found (Mahadih534 / bangla-tts-datasets)

Despite the name, the collection has only 2 datasets, and both are ASR datasets:

| Dataset | Size | Verdict |
|---|---|---|
| `SUST-CSE-Speech/banspeech` | 6.5 h, 13 domains (talk show, interview, drama, news, …) | **Too small to train on, but an excellent test set.** Bangladeshi speech, spontaneous domains, clean labels with English words written in Bangla script (same as our rule). We use it **only for testing**. |
| `SKNahin/open-large-bengali-asr-data` | ~5,000 h, 59 GB, 9 sources | **Not recommended.** Mostly *read* speech (Common Voice, OpenSLR, Shrutilipi, …), many wrong transcripts (it ships a `wer`/`is_better` column to filter them), license not stated. |

### 1.3 Everything else I checked

| Dataset | Type | Size | License | Verdict |
|---|---|---|---|---|
| **`ai4bharat/IndicVoices`**, Bengali | **Real** extempore + **conversational**. Two-person conversations were recorded **over an 8 kHz telephone bridge** | ~100+ h Bengali transcribed (96 parquet shards, ~40 GB) | CC-BY-4.0 (commercial use OK), gated: free click-through | ✅ **Main training set.** The only open Bangla data with real telephone conversations |
| **`SUST-CSE-Speech/SUBAK.KO`** | Real, **Bangladeshi** standard Bangla, mostly read in a studio, 12 h broadcast | 241 h | CC-BY-4.0 | ✅ **Clean mix** (~20%) so the model doesn't forget clean audio |
| `kawshikbuet17/bengali-telecom-customer-care-speech` | **Synthetic** TTS, 1 cloned voice, telecom scripts | 26.8 h | CC-BY-4.0 | ❌ for now. One synthetic voice teaches the model a TTS voice, not callers. Maybe later for vocabulary |
| `InfoBayAI/Bengali-Call-Center-Audio-Dataset-Single-Channel` | "Call center" | **2 sample files, no transcripts** | – | ❌ Advertisement for a paid dataset |
| `Suprio85/Bangla_Speech_Corpus` ("Bengali-Loop") | YouTube drama/audiobooks, auto-subtitles | ~155 h, 106 GB | Apache-2.0 (YouTube rights unclear) | ❌ Same kind of data as Lipi-Ghor (YouTube) + noisy auto-captions + broken loader |
| OOD-Speech (Bengali.AI) | Crowdsourced read + OOD test | 1,177 h | – | ❌ Not accessible on HF (401), mostly read speech |
| Appen BEN_ASR001 | **Real** Bangladesh telephony conversations, 8 kHz | 47 h | Paid | 💰 Closest to our calls if we ever buy data |

### 1.4 Recommendation

| Role | Dataset | Why |
|---|---|---|
| **Train** (≈80%) | **IndicVoices Bengali** | Real spontaneous/conversational speech, many speakers, part of it real 8 kHz phone calls, clean license, **not** in Lipi-Ghor (new data for the model) |
| **Train** (≈20%) | **SUBAK.KO** | Bangladeshi accent + protection against forgetting clean audio |
| **Augmentation** | Phone simulation on the IndicVoices clips | 8 kHz, 300–3400 Hz band, G.711 µ-law codec, line noise |
| **Test** (never trained on) | **BanSpeech** talk-show / interview / drama (clean + phone-simulated), TV news + audiobooks (clean); IndicVoices valid; **your own call recordings** | Bangladeshi, spontaneous, independent of the training data |

**Known limits:**
- IndicVoices Bengali speakers are from **India** (West Bengal and nearby states), so accent and some words
  differ from Bangladeshi callers. SUBAK.KO and the Bangladeshi test sets cover this. It is still the closest
  open match for *conversational phone* speech.
- IndicVoices is gated, so I could not see its exact column names or how many Bengali clips are 8 kHz
  phone recordings. The notebook **detects the columns automatically** and **prints** the sample rates and
  recording types, so you can see this in Part 4 before training.
- None of this data has our **hospital vocabulary** (doctor names, departments). Only our own calls can teach
  that. Don't expect this experiment to fix `রেজাউল হক` → `রেল ভাব`. Expect better handling of the **phone sound**
  and **conversational style**.

### 1.5 "The model is already trained on Lipi-Ghor. Do I need to include it?"

**Not for learning.** The model has already learned what Lipi-Ghor can teach it. Training on it again adds
almost nothing new, and the useful new information is in **new** data (conversation, phone audio).

**But some "old-style" data in the mix is useful as protection.** When you fine-tune only on new data, a
model can slowly **forget** what it was good at (*catastrophic forgetting*). The standard protection is to mix a
small share of data like the original training data back in ("rehearsal"). We do this with **SUBAK.KO at
~20%** instead of Lipi-Ghor, because Lipi-Ghor is 50 GB, gated, and has an unclear file layout. SUBAK.KO is
clean, transcribed and Bangladeshi, so it does the same job.

**How we know it worked:** the `test_bs_clean` score (TV news + audiobooks) before vs. after training, plus your
news/movie/AI voice files. If it gets worse by more than ~1–2 WER points, raise `CLEAN_MIX_RATIO`, lower the
learning rate, or (only then) add a Lipi-Ghor slice as rehearsal data.

**And never test on Lipi-Ghor.** The model has seen it, so its scores there are too optimistic.

---

## 2. The fine-tuning process in short

The notebook explains each step in detail (Part 1 of the notebook is a full explanation, no code). Summary:

```
 ┌─────────────── DATA ───────────────┐   ┌──────── TRAINING (repeat for every batch) ────────┐
 │ download parquet shards            │   │ 16 clips → collator: phone-sim (50%), pad, mask    │
 │ decode → mono → 16 kHz             │   │ model(audio, labels) → loss (cross-entropy)        │
 │ clean text (NFC, no tags/punct.)   │──►│ backprop → AdamW update (LR 3e-5, warm-up, decay)  │
 │ filter (1–20 s, ≤180 tokens, …)    │   │ every 250 steps: dev WER + checkpoint to Drive     │
 │ labels = tokens + [SEP] (no [CLS]) │   └───────────────────────┬────────────────────────────┘
 └────────────────────────────────────┘                           ▼
   BASELINE: score original model ──────────────►  FINAL: score best checkpoint, same test sets
                                                   → before/after table → save → test on PC
```

Key facts about **this** model, already checked against the real model:

| Fact | Consequence |
|---|---|
| Input = raw 16 kHz waveform, padded to a multiple of 320 samples | Our 8 kHz calls are upsampled to 16 kHz, in training and in use |
| Labels = text tokens + `[SEP]` (id 3), **without** `[CLS]` | Checked: this gives loss 1.78 vs 2.33 with `[CLS]` on a news clip, so it matches how the model was trained |
| The model adds `[CLS]` (id 2) as first decoder input itself | We only pass `labels` |
| The saved `generation_config.json` has **wrong** ids (1/2/2) | The notebook fixes it; the saved fine-tuned model has the correct one |
| Batched input with an attention mask gives the same output as one clip at a time | Batched training and evaluation are safe |
| Decoder has 194 positions | Labels ≤ 180 tokens, clips ≤ 20 s |

---

## 3. Step by step

### 3.1 One-time preparation (10 min)

1. **Hugging Face:** log in, open https://huggingface.co/datasets/ai4bharat/IndicVoices and **accept the
   conditions**. Create a **read** token at https://huggingface.co/settings/tokens.
2. **Google Drive:** make the folder `MyDrive/moonshine_finetune/sample-audio/` and upload the 5 files from
   this repo's `sample-audio/`. Optional but recommended: also upload reference transcripts (step 3.5).
3. **Colab:** File → Upload notebook → `moonshine_finetune_colab.ipynb`.
   Runtime → Change runtime type → **T4 GPU**. 🔑 Secrets (left sidebar) → add `HF_TOKEN` = your token,
   and switch on "Notebook access".

### 3.2 Quick test run (~10–15 min)

In the CONFIG cell set `QUICK_TEST = True` → *Runtime → Run all*. It uses 1 shard, ~0.5 h of data and 60
training steps. It is not meant to improve anything; it shows that login, downloads, disk, GPU, training,
evaluation and saving all work. **Look at Part 4's output** while it runs:
- Which text column was picked, and do the example transcripts look right (as spoken, no strange tags)?
  If another column looks better, set `IV_TEXT_COL`.
- The **native sample rates**: how many clips are `8000` (real phone recordings)?
- The short-list columns: is there one that marks conversations? You can use it later with `IV_KEEP`.

### 3.3 The real run (~1.5–2.5 h in total)

Set `QUICK_TEST = False` → *Runtime → Restart session and run all*. Default: up to 60 h of IndicVoices
(10 shards) + ~15 h SUBAK.KO, 3 epochs.

| Phase | Time on a T4 (estimate) |
|---|---|
| Download + prepare data | 20–40 min (CPU decoding, Colab has only 2 CPU cores) |
| Baseline evaluation | 5–10 min |
| Training | 45–90 min |
| Final evaluation + saving | 10–15 min |

These times are estimates; the notebook prints the real numbers. Keep the browser tab open: free Colab
disconnects idle sessions. **If it disconnects:** run all cells again. Data preparation repeats, but training
**resumes from the last checkpoint on Drive**.

### 3.4 Reading the results (notebook Part 8)

| Test set | What it tells you | Good result |
|---|---|---|
| `test_bs_spont_phone` | **Main score**: Bangladeshi spontaneous speech through a simulated phone line | WER clearly lower (≥ 3 points) |
| `test_bs_spont_clean` | Same clips, no phone sim: did it learn *conversation* or only *phone sound*? | Lower or equal |
| `test_iv_phone` | In-domain (same kind of data as training) | Lower. If only this improves, the model learned IndicVoices, not calls |
| `test_bs_clean` | **Forgetting check** (news, audiobooks) | Not worse than ~1–2 points |
| Your call recordings | The real target (read the transcripts; WER if you add references) | Fewer misheard words |

Baseline numbers for orientation, measured on the original model with 80 BanSpeech talk-show clips (greedy):
**clean 35.5% WER / 15.5% CER → phone-simulated 52.5% WER / 25.4% CER.** That ~17-point gap is what this
experiment tries to close.

### 3.5 Make your own call test measurable (recommended, 15 min)

Write the correct transcript of each call recording into `<audio name>.txt` next to the audio (both in
`sample-audio/` and on Drive), following the rules in the main plan (numbers as words, English words in
Bangla script, no punctuation). Then the notebook and `test_finetuned.py` print **WER/CER for your real
calls**, not just transcripts to eyeball. Two calls (~95 s) are a small test, but much better than none.

### 3.6 Test on your PC

1. Download `MyDrive/moonshine_finetune/<run name>/final_model.zip` and unzip it into
   `Fine-tune-moonshine/models/<run name>/`.
2. From the project root:
   ```
   .venv\Scripts\python Fine-tune-moonshine\test_finetuned.py --model Fine-tune-moonshine\models\run1_indicvoices_phone
   ```
   This writes `Fine-tune-moonshine/outputs/compare_<run>_<time>.md` with both transcripts, the speed (RTF) of
   both models, and WER/CER if reference `.txt` files exist. Speed should be unchanged (same model size).
3. The fine-tuned folder is a normal Hugging Face model folder, so `MoonshineBN("<folder>")` from
   `moonshine_base_bn/moonshine_bn.py` loads it directly.

---

## 4. What to do with the result

| Outcome | Next step |
|---|---|
| Phone + spontaneous test clearly better, clean test OK, own calls look better | The method works. Collect + label our own calls (main plan, steps 1–4), then fine-tune again **starting from this model** with our call data |
| Phone test better, own calls not | The open data differs too much from our calls (accent, vocabulary, real codec). Go straight to our own call data; this run still proves the pipeline |
| Clean test clearly worse | Forgetting: `CLEAN_MIX_RATIO` 0.2 → 0.35, `LEARNING_RATE` 3e-5 → 1e-5, or 2 epochs instead of 3 |
| Nothing changes | Try `LEARNING_RATE` 5e-5 and more data (20 shards). If still nothing, the gain must come from our own data |

**Follow-up experiments** (one change at a time, new `RUN_NAME` each time so the results stay separate):
1. Only conversation recordings (`IV_KEEP`), if the data has such a column
2. `PHONE_SIM_PROB` 0.5 → 0.8
3. 20 shards / 120 h (needs ~2× the time, maybe two Colab sessions)
4. Learning rate 1e-5 vs 3e-5 vs 5e-5

### Experiment log (fill in)

| Run | Data | LR / epochs | bs_spont_phone WER | bs_spont_clean WER | bs_clean WER | iv_phone WER | Own calls (WER or notes) |
|---|---|---|---|---|---|---|---|
| baseline | – | – | | | | | |
| run1_indicvoices_phone | IV 60 h + SUBAK.KO 20% | 3e-5 / 3 | | | | | |

---

## 5. Troubleshooting

| Problem | Fix |
|---|---|
| `No access to ai4bharat/IndicVoices` | Accept the conditions on the dataset page with the **same** account as the token; check the `HF_TOKEN` secret has notebook access |
| `CUDA out of memory` | `BATCH_SIZE = 8`, `GRAD_ACCUM = 4` |
| Loss is `nan` | `LEARNING_RATE = 1e-5` |
| Disk full | Fewer `IV_TRAIN_FILES`; delete `/content/work/raw` after Part 4 (the processed data is in `/content/work/cache`) |
| RAM crash during data preparation | Lower `IV_MAX_TRAIN_HOURS`; `TEST_CLIPS_PER_SPLIT = 200` |
| Session cut off during training | Run all cells again; training resumes from the last checkpoint on Drive |
| "You have reached your GPU usage limit" | Free Colab quota is used up; wait (usually hours to a day) or use Kaggle (30 GPU-h/week; the notebook runs there with paths changed) |
| Text column looks wrong (tags, Latin script) | Set `IV_TEXT_COL` to another column from the Part 4 printout |

---

## 6. Licenses

| Item | License | Note |
|---|---|---|
| Moonshine-Base-BN | MIT | |
| IndicVoices | CC-BY-4.0 | Commercial use allowed **with attribution** (AI4Bharat, IIT Madras). Keep a note in the model's docs |
| SUBAK.KO | CC-BY-4.0 | Attribution (SUST CSE) |
| BanSpeech | CC-BY-4.0 / CC-BY-NC-4.0 (the card is contradictory) | Used **only for testing**, not in the model |

## 7. How the notebook was checked

The notebook was run end to end on CPU with tiny settings (public BanSpeech files standing in for the gated
IndicVoices, 6 training steps): data preparation, baseline, training with checkpoints and dev WER, loading
the best checkpoint, before/after table, own-recordings report, saving + zip. `test_finetuned.py` was run
on the result. Its baseline transcripts of `sample-audio/` match the earlier comparison report word for word,
so evaluation decodes exactly like `moonshine_base_bn/test_model.py`. Not tested: the Colab-only cells (Drive
mount, secrets) and IndicVoices itself (gated). Part 4 of the notebook prints what it finds there.
