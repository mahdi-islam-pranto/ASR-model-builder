"""
STEP 1: download, inspect and prepare all data.

  python prepare_data.py --inspect     # download one training file, print columns/examples, save example wavs, stop
  python prepare_data.py               # build everything (skips if already built with the same settings)
  python prepare_data.py --force       # rebuild even if it exists

Output (data/processed/ or data/processed_quick/):
  train/             training clips from TRAIN_SOURCES (config.py), shuffled
  dev/               SUBAK.KO validation clips, half clean + half phone-simulated (picks the best checkpoint)
  tests/<name>/      test sets (never trained on), phone simulation already applied where the name says so
  examples/          wav files + spectrogram so you can LISTEN to the data and the phone simulation
  data_report.json   hours per source, dropped clips and why
  prep_settings.json the settings used; if config.py changes, the next run rebuilds automatically

What happens to each clip: decode -> mono -> 16 kHz -> clean the text -> filters -> tokenize -> stored as
16-bit audio on disk (Arrow), so the training set does not have to fit in RAM.
"""

import io
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np

import common as K
import config as C

DROPS = Counter()


def inspect_source(src, paths):
    """Print what the first training file contains and write example wavs (original + phone-simulated)."""
    import pyarrow.parquet as pq
    import soundfile as sf

    pf = pq.ParquetFile(paths[0])
    schema = pf.schema_arrow
    audio_col = K.find_audio_col(schema)
    str_cols = K.string_cols(schema)
    print(f"\n================ {src['name']}: what is in the data ================")
    print("all columns  :", schema.names)
    print("audio column :", audio_col, "| text column:", src["text_col"])
    if src["text_col"] not in schema.names:
        raise SystemExit(f"text_col '{src['text_col']}' is not a column of {paths[0]}")
    peek = pf.read_row_group(0, columns=str_cols).to_pandas()
    print("\n--- 5 example texts (raw -> cleaned training label) ---")
    for t in peek[src["text_col"]].head(5):
        print("  raw  :", str(t)[:160])
        print("  label:", K.normalize_text(t)[:160])
    for c in str_cols:
        vc = peek[c].value_counts()
        if 1 < len(vc) <= 30:
            print(f"\n[{c}]\n{vc.to_string()}")

    rows = pf.read_row_group(0, columns=[audio_col, src["text_col"]]).to_pylist()[:200]
    srs, durs = Counter(), []
    for r in rows:
        b = (r[audio_col] or {}).get("bytes")
        if b:
            info = sf.info(io.BytesIO(b))
            srs[info.samplerate] += 1
            durs.append(info.duration)
    print(f"\nnative sample rates (first {len(rows)} clips): {dict(srs)}   (8000 = telephone recordings)")
    print(f"duration: mean {np.mean(durs):.1f} s, max {np.max(durs):.1f} s")

    ex = C.processed_dir() / "examples"
    ex.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i, r in enumerate(rows[:3]):
        y, sr = K.decode_audio(r[audio_col]["bytes"])
        sf.write(ex / f"{src['name']}_{i}_{sr}Hz.wav", y, K.SR)
        sf.write(ex / f"{src['name']}_{i}_phone_sim.wav", K.phone_sim(y, rng), K.SR)
        (ex / f"{src['name']}_{i}.txt").write_text(str(r[src["text_col"]]), encoding="utf-8")
    print(f"\nexample audio written to {ex} (original + phone-simulated): listen to them!")
    print("=" * 68 + "\n")


def gen_rows(paths, audio_col, text_col, source, max_hours, keep=None, category_col=None, augment=1, cfg=""):
    """Yields training rows. The hours are spread evenly over the files (more speakers, more books)."""
    tokenizer = K.load_tokenizer()
    per_file = max_hours * 3600 / len(paths)
    for p in paths:
        total = 0.0
        for y, sr, text, cat in K.iter_clips([p], audio_col, text_col, keep, category_col):
            t, ids = K.check_clip(tokenizer, text, len(y) / K.SR)
            if t is None:
                DROPS[f"{source}: {ids}"] += 1
                continue
            yield {"audio": K.to_int16(y), "text": t, "labels": ids, "length": len(y), "native_sr": sr,
                   "source": source, "category": cat, "augment": augment}
            total += len(y) / K.SR
            if total >= per_file:
                break


def build(name, **kw):
    from datasets import Dataset
    kw["cfg"] = json.dumps(C.data_settings(), sort_keys=True)     # part of the cache key
    return Dataset.from_generator(gen_rows, gen_kwargs=kw, features=K.features(),
                                  cache_dir=str(C.processed_dir() / "_cache" / name))


def hours(ds):
    return float(np.sum(ds["length"])) / K.SR / 3600


def summary(ds, by):
    import pandas as pd
    df = pd.DataFrame({by: ds[by], "sec": np.asarray(ds["length"]) / K.SR})
    return df.groupby(by)["sec"].agg(clips="count", hours=lambda s: s.sum() / 3600, mean_sec="mean").round(2)


def load_rows(tokenizer, paths, audio_col, text_col):
    """All usable clips of (small) files: [(audio, text, native sr)]."""
    rows = []
    for y, sr, text, _ in K.iter_clips(paths, audio_col, text_col):
        t, _ = K.check_clip(tokenizer, text, len(y) / K.SR)
        if t is not None:
            rows.append((y, t, sr))
    return rows


def load_random_rows(tokenizer, paths, text_col, n, seed):
    """n random usable clips from (big) files, decoding only the clips that are picked."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.concat_tables([pq.read_table(p, columns=["audio", text_col]) for p in paths])
    texts = table.column(text_col).to_pylist()
    audio = table.column("audio")
    rows = []
    for i in np.random.default_rng(seed).permutation(len(texts)):
        b = (audio[int(i)].as_py() or {}).get("bytes")
        if not b:
            continue
        y, sr = K.decode_audio(b)
        t, _ = K.check_clip(tokenizer, texts[int(i)], len(y) / K.SR)
        if t is not None:
            rows.append((y, t, sr))
            if len(rows) >= n:
                break
    return rows


def simulate(row, seed):
    """(audio, text, sr) -> the same clip through a simulated phone line (fixed seed = identical every run)."""
    y, t, sr = row
    return (K.phone_sim(y, np.random.default_rng(seed)) if sr > 8000 else y), t, sr


def sample_rows(rows, n, seed, phone):
    idx = [int(i) for i in np.random.default_rng(seed).permutation(len(rows))][:n]
    return [simulate(rows[i], seed * 100003 + i) if phone else rows[i] for i in idx]


def save_rows(tokenizer, rows, path, source):
    from datasets import Dataset
    Dataset.from_dict({
        "audio": [K.to_int16(y) for y, _, _ in rows], "text": [t for _, t, _ in rows],
        "labels": [tokenizer(t, add_special_tokens=False)["input_ids"] + [K.EOS_ID] for _, t, _ in rows],
        "length": [len(y) for y, _, _ in rows], "native_sr": [sr for _, _, sr in rows],
        "source": [source] * len(rows), "category": [""] * len(rows), "augment": [0] * len(rows),
    }, features=K.features()).save_to_disk(str(path))
    print(f"  {path.name:24s} {len(rows):5d} clips  {sum(len(r[0]) for r in rows) / K.SR / 3600:.2f} h")


def phone_sim_picture(rows, ex_dir):
    """Spectrogram: clean vs phone-simulated vs a real call (if one is in MY_AUDIO_DIR)."""
    import librosa
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import soundfile as sf

    y_clean = rows[0][0]
    y_phone = K.phone_sim(y_clean, np.random.default_rng(0))
    sf.write(ex_dir / "banspeech_clean.wav", y_clean, K.SR)
    sf.write(ex_dir / "banspeech_phone_sim.wav", y_phone, K.SR)
    panels = [("clean", y_clean), ("phone-simulated", y_phone)]
    my = Path(C.MY_AUDIO_DIR)
    real = sorted(my.glob("*call*")) if my.is_dir() else []
    if real:
        y_real, _ = librosa.load(str(real[0]), sr=K.SR, offset=10, duration=6)
        panels.append(("real call: " + real[0].name, y_real))
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 3.2))
    for ax, (title, y) in zip(np.atleast_1d(axes), panels):
        S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=512)), ref=np.max)
        ax.imshow(S, origin="lower", aspect="auto", cmap="magma", extent=[0, len(y) / K.SR, 0, K.SR / 2])
        ax.axhline(3400, color="cyan", lw=0.8, ls="--")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("seconds")
        ax.set_ylabel("Hz")
    plt.tight_layout()
    plt.savefig(ex_dir / "phone_sim_spectrogram.png", dpi=110)
    plt.close(fig)
    print(f"  spectrogram + wavs: {ex_dir}")


def main(args):
    import pyarrow.parquet as pq
    from datasets import concatenate_datasets
    from huggingface_hub import list_repo_files

    out = C.processed_dir()
    settings_file = out / "prep_settings.json"
    settings = C.data_settings()
    if not args.force and not args.inspect and settings_file.exists():
        if json.loads(settings_file.read_text(encoding="utf-8")) == settings:
            print(f"Data already prepared with these settings in {out}. Use --force to rebuild.")
            return
        print("Data settings changed since the last build -> rebuilding.")
    t_start = time.time()

    if args.inspect:
        src = C.TRAIN_SOURCES[0]
        inspect_source(src, K.download(src["repo"], src["files"][:1]))
        return

    for d in ("train", "dev", "tests", "_cache"):
        shutil.rmtree(out / d, ignore_errors=True)
    settings_file.unlink(missing_ok=True)
    tokenizer = K.load_tokenizer()
    report = {"dropped": {}}

    print("\n[1/4] Training data")
    parts = []
    for n, src in enumerate(C.TRAIN_SOURCES):
        print(f"\n--- {src['name']}: {len(src['files'])} files from {src['repo']}, up to {src['hours']} h")
        paths = K.download(src["repo"], src["files"])
        if n == 0:
            inspect_source(src, paths)
        DROPS.clear()
        t0 = time.time()
        ds = build(src["name"], paths=paths, audio_col=K.find_audio_col(pq.ParquetFile(paths[0]).schema_arrow),
                   text_col=src["text_col"], source=src["name"], max_hours=src["hours"], keep=src.get("keep"),
                   category_col=src.get("category_col"), augment=1 if src.get("phone_sim", True) else 2)
        print(f"  built in {(time.time() - t0) / 60:.1f} min: {len(ds)} clips, {hours(ds):.1f} h")
        if src.get("category_col"):
            print(summary(ds, "category"))
        print(summary(ds, "native_sr"))
        print("  dropped:", dict(DROPS))
        report["dropped"].update(DROPS)
        if hours(ds) < 0.9 * src["hours"]:
            print(f"  NOTE: only {hours(ds):.1f} of {src['hours']} h. Add more files to this source for the full amount.")
        parts.append(ds)
    train = concatenate_datasets(parts).shuffle(seed=C.SEED)
    train.save_to_disk(str(out / "train"))
    report["train"] = summary(train, "source").reset_index().to_dict("records")
    report["train_by_native_sr"] = summary(train, "native_sr").reset_index().to_dict("records")
    report["train_hours"] = round(hours(train), 2)
    report["train_clips"] = len(train)
    print(f"\n  TRAIN: {len(train)} clips, {hours(train):.1f} h -> {out / 'train'}")
    del train, parts
    shutil.rmtree(out / "_cache", ignore_errors=True)

    print("\n[2/4] Dev set (SUBAK.KO validation: half clean, half phone)")
    dev = load_random_rows(tokenizer, K.download(C.SUBAKKO_REPO, C.SUBAKKO_VALID_FILES), "transcription",
                           C.DEV_CLIPS, seed=1)
    half = len(dev) // 2
    dev = dev[:half] + [simulate(r, 1000 + i) for i, r in enumerate(dev[half:])]
    save_rows(tokenizer, dev, out / "dev", "dev_subakko_mixed")
    del dev

    print("\n[3/4] Test sets")
    sk_test = load_random_rows(tokenizer, K.download(C.SUBAKKO_REPO, C.SUBAKKO_TEST_FILES), "transcription",
                               C.TEST_CLIPS_PER_SPLIT, seed=2)
    bs_files = {f.split("/")[-1].split("-")[0]: f for f in list_repo_files(C.BANSPEECH_REPO, repo_type="dataset")
                if f.startswith("data/") and f.endswith(".parquet")}
    cache = {}

    def banspeech(splits, phone, seed=7):
        rows = []
        for k, s in enumerate(splits):
            if s not in cache:
                cache[s] = load_rows(tokenizer, K.download(C.BANSPEECH_REPO, [bs_files[s]]), "audio", "transcription")
            rows += sample_rows(cache[s], C.TEST_CLIPS_PER_SPLIT, seed=seed + k, phone=phone)
        return rows

    tests = {
        "test_subakko_clean": sk_test,                                                  # standard Bangla, studio
        "test_subakko_phone": [simulate(r, 5000 + i) for i, r in enumerate(sk_test)],   # the same clips, phone line
        "test_bs_spont_phone": banspeech(C.BANSPEECH_SPONT, phone=True),                # conversation, phone line
        "test_bs_spont_clean": banspeech(C.BANSPEECH_SPONT, phone=False),
        "test_bs_medicine_phone": banspeech(C.BANSPEECH_MEDICINE, phone=True),          # medical words, phone line
        "test_bs_clean": banspeech(C.BANSPEECH_CLEAN, phone=False),                     # news + audiobooks
    }
    for name, rows in tests.items():
        save_rows(tokenizer, rows, out / "tests" / name, name)
    report["tests"] = {k: len(v) for k, v in tests.items()}

    print("\n[4/4] Phone simulation example")
    phone_sim_picture(tests["test_bs_spont_clean"], out / "examples")

    (out / "data_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    settings_file.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")   # marks "done"
    if C.DELETE_RAW_AFTER_PREP:
        shutil.rmtree(C.raw_dir(), ignore_errors=True)
        print(f"deleted downloads in {C.raw_dir()} (DELETE_RAW_AFTER_PREP)")
    print(f"\nData ready in {out} ({(time.time() - t_start) / 60:.1f} min). Report: {out / 'data_report.json'}")


def parser():
    p = K.base_parser("Step 1: download + prepare data")
    p.add_argument("--inspect", action="store_true", help="only download one training file + show what is in it")
    p.add_argument("--force", action="store_true", help="rebuild even if prepared data exists")
    return p


if __name__ == "__main__":
    a = parser().parse_args()
    K.setup(a, "prepare_data")
    main(a)
