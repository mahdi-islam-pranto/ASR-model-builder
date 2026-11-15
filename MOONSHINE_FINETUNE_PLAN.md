# Moonshine-Base-BN: fine-tuning plan for call-center recordings

**Goal:** make `Sanjidh090/moonshine-base-bn` (61.5M parameters, MIT license) accurate on our own call-center recordings (8 kHz phone audio, hospital calls, Bangla with English words mixed in) **without getting worse on clean audio**, where it already does well.

This is a plan to read and discuss. No code has been written yet. Each step ends with a **deliverable**, so we can do the steps one at a time.

---

## 0. Where we are now (baseline)

Source: [comparison/comparison_20261112_165503.md](comparison/comparison_20261112_165503.md) and [moonshine_base_bn/outputs/results_20261112_165326.txt](moonshine_base_bn/outputs/results_20261112_165326.txt).

### The test audio

| File | Length | Format |
|---|---|---|
| `hospital inbound call center.mp3` | 51.0 s | **8 kHz, mono, MP3** |
| `hospital outbound call center.mp3` | 43.7 s | **8 kHz, mono, MP3** |

Mono means **agent and customer are mixed on one channel**. MP3 means the audio has been compressed twice: once by the phone line and again by the MP3 encoder. Both facts matter for data preparation (Step 3).

### What Moonshine gets wrong on these calls

I read the Moonshine transcripts next to the other models' outputs, since they often get different parts right. The errors fall into these groups:

| # | Error type | Example (model output → probable correct text) | What fixes it |
|---|---|---|---|
| A | **Hold music, ring tone or laughter turned into tags or words** | `মিউজিক মিউজিক দিদি …`, `সশব্দ হাসি` | Speech detection (VAD) before ASR + a tagging rule in the training data |
| B | **Words misheard because of the phone line** | `ছয়টার` → `ফাটার`, `সাড়ে` → `সারা`, `সুবাদে` → `শুভ্রায়`, `হাসপাতালে` → `আসার পাতলা` | Fine-tuning on real phone audio (main fix) + phone-audio simulation |
| C | **Names and domain words** | `ডাক্তার রেজাউল হক` → `ডাক্তার রেল ভাব`, `সিরিয়াল` → `ক্লিয়ার`, `মাহমুদা` → `মাহমুদুল / মোহাম্মদা`, opening hospital/branch name garbled | Fine-tuning + a list of known names (doctors, departments, branches) for post-correction |
| D | **Inconsistent number style** | `16` (digits) while other models write `ষোলো` | One rule in the transcription guide (Step 4) + a number formatting step after ASR |
| E | **Call openings are weak** | First 3–5 s of both calls are mostly wrong in every model | Usually the agent's fast scripted greeting over line noise. More data for openings + VAD |
| F | **Words lost at chunk borders** | Fixed 20 s chunks in `test_model.py` can split a word in half | Split audio at pauses (VAD) instead of every 20 s, with no training needed |

**What already works:** the full content of the inbound call (hospital, patient "হৃদয়", complaint question, `আসসালামু আলাইকুম`), most of the outbound call, and only 3–4% repeated words, so it does not get stuck in loops like FastConformer did. The model's knowledge of Bangla is good. What it lacks is **phone-quality audio and our vocabulary**, and fine-tuning is the right tool for both.

### Facts about the model that affect the plan

- **Input:** raw waveform at **16 kHz**. Our 8 kHz audio must be **upsampled to 16 kHz** at load time. This does not add information, but it is what the model expects.
- **Decoder limit:** `max_position_embeddings = 194`. One training segment can have **at most about 190 tokens**, roughly 140 Bangla words. Keeping segments **≤ 20 s** stays well inside this.
- **Tokenizer:** BanglaBERT WordPiece (32k vocab). I checked it: `ষোলো`, `নাম্বার`, `সিরিয়াল`, `ডাক্তার`, `রেজাউল`, `আসসালামু`, `মিউজিক` are each **one token**. **We do not need to change the vocabulary.** We only need consistent transcripts.
- **Training data it came from:** Lipi-Ghor (882 h, mostly clean, multi-domain). The `মিউজিক` and `সশব্দ হাসি` outputs show that this dataset labels non-speech with those tags.
- **Special tokens:** start = 2 (`[CLS]`), end = 3 (`[SEP]`), pad = 0.

---

## 1. Decisions to make before starting

These change how the data is made, so it is cheaper to decide them now than to re-label later.

| # | Question | My recommendation |
|---|---|---|
| D1 | **Commercial use?** | Assumed yes, so only MIT/Apache tools and data. Do **not** use the Hishab (non-commercial) models to make training labels. |
| D2 | **Can the phone system record in stereo** (agent on left, customer on right)? | Ask the PBX/telephony team. If yes, record **new calls in stereo**. It makes speaker separation free and accurate. Old mono calls are still usable. |
| D3 | **Numbers: words or digits?** | **Words in the transcript** (`ষোলো`, `সাড়ে পাঁচটা`). The model hears words, and a small formatting step (Step 8) turns them into digits (`16`, `5:30`) for reports. |
| D4 | **English words** (patient, serial, doctor, ICU, CT scan) | **Bangla script, as said** (`পেশেন্ট`, `সিরিয়াল`, `সিটি স্ক্যান`). This matches how the model already writes them. Keep a fixed spelling list. |
| D5 | **Non-speech** (music, ring tone, laughter, noise) | Remove most of it with VAD. In training text, use a **small fixed tag set** (Step 4), or none at all. |
| D6 | **Dialect** | Write what was **said**, not standard Bangla (`যাইতে`, `থাইকে`). The analysis step can handle variants. |
| D7 | **Privacy** | Calls contain patient names, phone numbers and health details. Decide who can listen, where audio is stored, and that nothing goes to public Hugging Face repos. Training on Colab with a private Google Drive is OK if company policy allows; otherwise rent a GPU or use an in-house machine. |

---

## 2. Step 1: build the test set first

You cannot improve what you do not measure. Before collecting training data, build a **held-out test set** that is **never** used for training.

**Size:** 1–2 hours of call audio (about 60–150 calls), chosen to cover:
- inbound **and** outbound calls
- different agents (at least 8–10 if possible) and customers (male/female, young/old)
- different call types: appointments/serial, complaints, reports, billing, follow-up calls
- noisy calls, short calls, calls with hold music
- different dates or weeks (phone line quality changes)

**Rules:**
- Split **by call**, never by segment. A segment from a test call must never appear in training.
- Ideally, the test set agents are **not** in the training data, or at least keep a separate "unseen agent" subset, so we know whether the model works on new staff.
- Transcribe by hand following the guide in Step 4, with a second person checking.

**Also keep a "clean" regression set:** the 3 existing clean files (AI voice, news, movie) + ~1 hour of Lipi-Ghor test data. This catches the model getting worse on clean audio.

**Run the baseline:** put the reference text next to each file (`<name>.txt`) and run `compare_models.py`. That gives real WER/CER for every model **before** fine-tuning.

**Measures to record:**
| Measure | Why |
|---|---|
| WER and CER on the call test set | Main score. CER matters a lot for Bangla because small spelling differences inflate WER |
| **Key-term accuracy**: % of doctor names, department names, numbers/times, hospital/branch names that come out right | This is what the call analysis actually depends on |
| Hallucination on non-speech: words produced on music/silence segments | Should be close to 0 |
| WER/CER on the clean regression set | Must not get noticeably worse |
| RTF on your CPU | Must stay usable (now 0.21) |

**Deliverable:** `data/test/` with audio + reference text, and a baseline score table.

---

## 3. Step 2: collect and clean the raw calls

1. **Export recordings** from the call system together with metadata: call ID, date, direction (in/out), agent ID, length, queue or department. Metadata is what we use to balance the data and split it correctly.
2. **Keep the original format** (8 kHz, as saved). Do not "improve" the audio. The model must learn the real phone sound.
3. **Filter out** calls that are unusable (only IVR menu, only ringing, under 5 s, dead air).
4. **Handle IVR/hold audio:** the same recorded IVR prompt or hold music appears in hundreds of calls. If it all goes into training, the model memorises it. Keep a few examples of each, drop the rest.
5. **Store safely** (D7). Use a call-ID naming scheme, with no patient names in file names.

**How much data:**
| Amount of hand-checked call audio | What to expect (rough, from typical fine-tuning results) |
|---|---|
| 5–10 h | Clear improvement on common phrases and the main domain words; good for a first trial |
| 20–50 h | Solid improvement across agents and call types. **Target for the first real model** |
| 100 h+ | Close to the best this model size can do |

These are experience-based estimates, not guarantees. The test set from Step 1 tells us the real numbers.

**Deliverable:** `data/raw/` with calls + a `calls.csv` metadata sheet.

---

## 4. Step 3: cut calls into training segments

Moonshine was trained on short segments, not 5-minute calls. Each call must be cut into pieces of **2–20 seconds**, cut at pauses, never mid-word.

**Pipeline (per call):**
1. **Load** → mono, float, **resample 8 kHz → 16 kHz** (same as at inference time).
2. **(If stereo)** split into agent and customer channels and process each separately.
3. **Speech detection with Silero VAD** (MIT, runs on CPU): find the speech regions, drop long silence and music.
4. **Merge** nearby speech regions into segments of **2–20 s** (target 5–15 s), cutting at the longest pauses. Keep ~0.2 s of padding on each side.
5. **Save** each segment as 16 kHz WAV with a manifest row:

```json
{"audio": "segments/call_00123_0007.wav", "duration": 8.4, "text": "", "call_id": "call_00123",
 "direction": "outbound", "agent_id": "A07", "channel": "mono", "split": "train"}
```

The same VAD-based cutting is then also used at inference time (Step 8), so training and real use see the same kind of segments.

**Deliverable:** `data/segments/` + `manifest_*.jsonl`, plus a script that does this for any new batch of calls.

---

## 5. Step 4: transcribe the segments (the main work)

### 5.1 Workflow: model draft + human correction

1. Run current Moonshine on every segment to get a **draft transcript** (beam 5).
2. People **listen and correct** the draft in an annotation tool. **Label Studio** (free, self-hosted, so the audio stays in-house) has a ready "audio transcription" template.
3. A second person **checks 10%** of the work. Measure the CER between the two versions: under ~5% CER means the guide is clear. Higher means the guide needs fixing.
4. Mark segments as **unusable** (overlapping talk you can't separate, unintelligible) instead of guessing.

Correcting a draft is usually **2–3× faster** than typing from scratch. As a rough guide, budget **4–6 hours of human work per hour of audio** for correction + review. The model draft gets better after each fine-tuning round (Step 9), so later batches go faster.

> Do not train on unchecked model drafts. Training on its own mistakes teaches the model that its mistakes are correct.

### 5.2 Transcription guide (write it once, everyone follows it)

This is the most important document in the project. Inconsistent transcripts are the most common reason fine-tuning disappoints.

| Topic | Rule (proposal) | Example |
|---|---|---|
| Write what is **said** | Including dialect and informal forms | `যাইতে হবে`, `থাইকে` |
| Numbers | **Words**, as spoken | `ষোলো নাম্বার`, `সাড়ে পাঁচটা`, `শূন্য এক সাত…` |
| English words | Bangla script, fixed spelling from the term list | `পেশেন্ট`, `সিরিয়াল`, `রিপোর্ট`, `আইসিইউ` |
| Names | Correct spelling from the doctor/staff/branch list | `ডাক্তার রেজাউল হক` |
| Greetings | Fixed spelling | `আসসালামু আলাইকুম`, `ওয়ালাইকুম আসসালাম` |
| Fillers | Keep short ones that are clearly spoken (`হ্যাঁ`, `জি`, `আচ্ছা`, `হুম`); skip "uh/um" sounds | |
| Repeated words | Write them if actually said | `হ্যাঁ হ্যাঁ` |
| Punctuation | **None**; it makes WER worse and the model doesn't output it | |
| Non-speech | Either **no tag** (VAD should remove most) or only these: `[মিউজিক]`, `[হাসি]`, `[অস্পষ্ট]` | Decide once (D5) |
| Cut-off word at segment edge | Write it if understandable, else leave it out | |
| Unicode | Always save as **NFC**; use one form of `য়`, `ড়`, `ঢ়` (precomposed vs. base+nukta) | Done automatically by a cleanup script |

**Term lists to collect from the hospital** (needed for transcription *and* for post-correction in Step 8):
- doctor names + departments (`নিউরোলজি`, `কার্ডিওলজি`, …)
- hospital and branch names (the outbound call opens with the hospital/branch name, which every model garbled)
- common services, tests, medicines, packages
- fixed agent scripts (greeting, closing, complaint/feedback question)

**Deliverable:** `TRANSCRIPTION_GUIDE.md`, term lists (`terms/doctors.txt`, …), and the first batch of checked transcripts.

---

## 6. Step 5: build the training dataset

1. **Text cleanup script:** NFC normalisation, character unification, remove punctuation, check against the tag set, drop empty or unusable rows.
2. **Length checks:** drop segments < 1 s or > 20 s, and any whose text is over **~180 tokens** (decoder limit 194). Also drop segments where text length doesn't fit audio length (e.g. 10 s of audio with 1 word), which usually means a labeling mistake.
3. **Split by call** (and keep the Step 1 test set fully separate):
   - `train` ~90%, `dev` ~10% (used during training to pick the best checkpoint)
   - `test` = Step 1 set, untouched
4. **Add some clean data to prevent forgetting:** mix in Lipi-Ghor training data at about **20–30%** of each batch. This keeps the model good on clean audio. (Check Lipi-Ghor's license before using it in a commercial product.)
5. Save as a Hugging Face `datasets` folder (or Parquet) so it loads quickly on Colab.

**Deliverable:** `data/hf_dataset/{train,dev,test}` + a short data report (hours per split, agents, call types, avg segment length).

---

## 7. Step 6: augmentation, especially phone-audio simulation

Augmentation creates extra training variety from the data we have. For us, the most useful kind is **making clean audio sound like a phone call**. This can start **before** any call data is labelled.

**Phone simulation** (applied on the fly to clean Lipi-Ghor audio):
- downsample to 8 kHz and back up to 16 kHz (removes everything above 4 kHz)
- band-pass ~300–3400 Hz (phone band)
- **G.711 µ-law / A-law** encode-decode (standard phone codec), sometimes **AMR-NB** or **GSM** for mobile callers
- **MP3 re-encode at low bitrate** (our recordings are MP3)
- random gain, light clipping, line noise, office/call-center background noise, short dropouts (packet loss)

**General augmentation on real call segments (light):**
- speed perturbation 0.9× / 1.0× / 1.1×
- small volume changes, light extra noise

Tools: `audiomentations` (MIT), `torchaudio` codec functions, `ffmpeg` for AMR/GSM/MP3.

> Augmentation helps, but it **does not replace real call data**. Real calls have things simulation doesn't: talking over each other, call-center phrasing, regional accents, real line problems.

**Deliverable:** an augmentation module + a few listening samples to check it sounds like our real calls.

---

## 8. Step 7: training

### 8.1 Two stages

| Stage | Data | Purpose | Needs labelled calls? |
|---|---|---|---|
| **A: phone adaptation** (optional but cheap) | Lipi-Ghor (or a part of it) with **phone simulation** applied | Get the model used to 8 kHz-band phone audio | **No** — can run right away |
| **B: call-center fine-tune** | Our checked call segments + 20–30% clean Lipi-Ghor | Learn the real phone sound, hospital vocabulary, agent scripts, accents | Yes |

Stage A then Stage B usually gives better results than Stage B alone, especially when we only have 5–20 h of call data. If the test set shows Stage A doesn't help, we drop it.

### 8.2 Setup

- **Framework:** Hugging Face `transformers` `Seq2SeqTrainer` (the model loads with `AutoModelForSpeechSeq2Seq`, as in our test script).
- **Hardware:** free **Colab T4 (16 GB)** is enough for a 61M model. Kaggle (2× T4, 30 h/week free) is an alternative. Training on our CPU is **not practical**. Inference stays on CPU.
- **Input:** 16 kHz waveform, padded to a multiple of 320 samples, with an attention mask for padding in batches.
- **Labels:** tokenizer IDs of the text followed by end token `3`. The trainer adds start token `2` itself. Important pitfall: the BERT tokenizer adds `[CLS]` at the front by default, which **must be removed** from the labels, or the model learns to output it twice. Padding in labels → `-100`, so it's ignored in the loss. **Before training,** we check this exactly against how the original model was trained by looking at its outputs (first and last token IDs).

### 8.3 Starting hyperparameters (to tune on the dev set)

| Setting | Start value | Note |
|---|---|---|
| Learning rate | 1e-5 to 3e-5 (Stage B); up to 5e-5 (Stage A) | Low, because we adapt an already trained model |
| Schedule | Linear or cosine, 5–10% warm-up | |
| Effective batch | ~32–64 segments | Use gradient accumulation on T4 |
| Precision | fp16 (T4 doesn't support bf16) | If loss becomes NaN → fp32 or lower LR |
| Epochs | 3–10, **early stopping** on dev WER | Small data overfits fast |
| Grouping | Sort batches by length (`group_by_length`) | Faster, less padding |
| Evaluate | Every N steps: dev WER/CER with **the same decoding settings as production** (beam, no-repeat-ngram) | Loss alone is misleading for this model (the model card explains this) |
| Freezing | First try: train everything. If overfitting with little data: freeze the decoder's embedding layer or first encoder layers | |

### 8.4 Experiments to compare (one variable at a time)

1. Stage B only
2. Stage A → Stage B
3. With vs. without clean-data mixing (watch the clean regression set)
4. Amount of call data: 10 h vs. 20 h vs. all (shows whether labelling more is worth it)

Log every run: data version, settings, dev/test WER/CER, key-term accuracy, clean-set WER. TensorBoard or a simple CSV is enough.

**Deliverable:** training notebook (Colab) + a script version, best checkpoint, results table.

---

## 9. Step 8: inference pipeline (part of it gives gains without training)

These changes help the current model **today** and the fine-tuned model later. Several can be done in parallel with data labelling.

1. **VAD-based cutting** (Silero) instead of fixed 20 s chunks. Removes hold music and silence (error A), and stops words being cut in half (error F).
2. **(If stereo)** transcribe agent and customer separately, giving "who said what" for free. If mono: add speaker separation (diarization) later. That's a separate project and not needed for the first model.
3. **Decoding settings:** keep beam 5 + `no_repeat_ngram_size=3` + `repetition_penalty=1.2`, but **check `no_repeat_ngram_size` on the test set**: it can wrongly block real repeats (`ঠিক আছে ঠিক আছে`). Also try beam 1 vs 5 for the speed/quality trade-off.
4. **Number formatting (inverse text normalisation):** rule-based conversion of `ষোলো নাম্বার` → `16 নম্বর`, `সাড়ে পাঁচটা` → `5:30` for reports. Keep the raw text too.
5. **Term correction:** match words against the term lists from Step 4 with fuzzy matching, e.g. a doctor-name list fixes `রেজাউল হক` variants. Only replace when it's a close match, to avoid creating errors.
6. **Confidence score per segment** (from beam scores). Low-confidence segments get flagged in reports and are picked for the next labelling round (Step 10).
7. **Speed on CPU:** export to ONNX + int8 quantisation once the model is final. Should give a clear speed-up, but needs a WER re-check after quantisation.

**Deliverable:** `pipeline/` with `vad → asr → format → terms` steps that run on a full call file.

---

## 10. Step 9: evaluate and decide

On the **Step 1 test set**, compare: original Moonshine vs. each fine-tuned version (and the other models from the earlier comparison, for reference).

**Accept a new model only if:**
- call WER **and** CER clearly improve (not just within noise; the test set must be large enough, which is why we want 1–2 h)
- key-term accuracy improves (numbers, doctors, hospital names)
- no-speech hallucination is not worse
- clean regression set is not noticeably worse (e.g. no more than ~1–2 WER points worse)
- RTF on CPU is unchanged (the model size doesn't change, so it should be)

Also **listen to 20–30 random test segments** and read the outputs. Numbers can hide problems such as tag spam or dropped words.

**Deliverable:** evaluation report (like the comparison file, but with WER/CER).

---

## 11. Step 10: keep improving (loop)

After the first fine-tuned model:
1. Run it on new calls.
2. Pick segments to label next: **low confidence**, **new call types**, **new agents**, **segments where the term matcher found nothing**.
3. Correct them (faster now, because the drafts are better), add to training, retrain.
4. Re-score on the same test set. Refresh the test set only rarely (e.g. every few months, adding new agents/call types), and record which version was used.

---

## 12. Proposed folder layout

```
ASR-models/
├── moonshine_finetune/
│   ├── TRANSCRIPTION_GUIDE.md
│   ├── terms/                     # doctors.txt, departments.txt, branches.txt, english_words.txt
│   ├── scripts/
│   │   ├── segment_calls.py       # Step 3: load, resample, VAD, cut, manifest
│   │   ├── draft_transcripts.py   # Step 4: Moonshine drafts for Label Studio
│   │   ├── import_labels.py       # Label Studio export → manifest
│   │   ├── clean_text.py          # Unicode/tag/punctuation cleanup
│   │   ├── build_dataset.py       # Step 5: checks, splits, HF dataset
│   │   ├── augment.py             # Step 6: phone simulation + light augmentation
│   │   ├── train.py               # Step 7
│   │   └── evaluate.py            # Steps 1/9: WER, CER, key terms, hallucination, RTF
│   ├── notebooks/
│   │   └── train_moonshine_colab.ipynb
│   └── data/                      # git-ignored: raw/, segments/, hf_dataset/, test/
└── pipeline/                      # Step 8: VAD → ASR → number formatting → term correction
```

Audio, transcripts and checkpoints stay **out of git** (privacy + size).

---

## 13. Rough order and timeline

| Phase | Work | Can run in parallel with |
|---|---|---|
| 1 | Decisions D1–D7, collect term lists, write transcription guide | |
| 2 | Segmentation script + Label Studio setup; label the **test set** (1–2 h) | VAD pipeline (Step 8.1) |
| 3 | Baseline scores on the test set for all models | Stage A training (phone simulation, no labels needed) |
| 4 | Label first **10 h** of training data | |
| 5 | First Stage B fine-tune + evaluation: **first real result** | Continue labelling to 20–50 h |
| 6 | Retrain with more data, tune settings, add number formatting + term correction | |
| 7 | ONNX/int8 export, plug into the analysis pipeline | |

The **labelling speed** sets the timeline, not the training (one training run on a T4 for ~10–50 h of data should take a few hours at most).

---

## 14. Risks and how we handle them

| Risk | Handling |
|---|---|
| Inconsistent transcripts | One guide, term lists, 10% double-checking, automatic cleanup script |
| Model forgets clean audio | Mix 20–30% clean data, check the clean regression set every run |
| Overfitting to a few agents or the IVR script | Balance by agent, limit repeated IVR/script audio, early stopping, test on unseen agents |
| Test data leaking into training | Split by call ID, test set stored in its own folder, a check script for overlap |
| Too little data at first | Stage A phone simulation, start with 10 h, measure the gain per added hour |
| Privacy of patient data | Self-hosted labelling, private storage, no public uploads, access control (D7) |
| Code/tool licences | MIT/Apache only (Moonshine, Silero VAD, Label Studio, audiomentations). Check Lipi-Ghor's licence before commercial use |
| Mono audio with both speakers mixed | Works for transcription; ask for stereo recording (D2) for speaker-level analysis |

---

## 15. What I need from you to start

1. Answers to **D1–D7** (especially commercial use, stereo recording, number style).
2. Access to **more call recordings** (how many hours exist, and how they can be exported with metadata).
3. **Term lists** from the hospital: doctors, departments, branches, services.
4. Who will **label** (in-house staff? how many people, hours per week?) — this sets the timeline.
5. **Training hardware:** Colab/Kaggle OK under your privacy rules, or a local/rented GPU?

Once you've decided, a good first piece of work is **Step 3 (segmentation script) + Step 8.1 (VAD-based inference)**. Both are useful immediately: VAD improves the current model's output on hold music and chunk borders, and segmentation is needed for all labelling.
