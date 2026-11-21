"""
STEP 1: download, inspect and prepare all data.

  python prepare_data.py --inspect     # download a shard, print columns/examples/sample rates, save example wavs, stop
  python prepare_data.py               # build everything (skips if already built with the same settings)
  python prepare_data.py --force       # rebuild even if it exists

Output (data/processed/ or data/processed_quick/):
  train/             training clips: IndicVoices (augment=1) + SUBAK.KO clean mix (augment=2), shuffled
  dev/               IndicVoices valid clips, phone-simulated (picks the best checkpoint during training)
  tests/<name>/      test sets (never trained on), phone simulation already applied where the name says so
  examples/          wav files + spectrogram so you can LISTEN to the data and the phone simulation
  data_report.json   hours per source, dropped clips and why, chosen columns
  prep_settings.json the settings used; if config.py changes, the next run rebuilds automatically

What happens to each clip: decode -> mono -> 16 kHz -> clean the text -> filters -> tokenize -> stored as
16-bit audio on disk (Arrow), so the training set does not have to fit in RAM.
"""

import io
import json
import shutil
import time
from collections import Counter

import numpy as np

import common as K
import config as C

DROPS = Counter()


def inspect_iv(iv_paths):
    """Print what the IndicVoices files contain, so the right columns can be chosen. Returns (audio, text, category) columns."""
    import pandas as pd
    import pyarrow.parquet as pq
    import soundfile as sf

    pf = pq.ParquetFile(iv_paths[0])
    schema = pf.schema_arrow
    audio_col = K.find_audio_col(schema)
    str_cols = K.string_cols(schema)
    text_col = C.IV_TEXT_COL or next(
        (c for c in ["text", "normalized", "verbatim", "transcription", "transcript", "sentence"] if c in str_cols), None)
    if not text_col:
        raise SystemExit(f"Could not auto-detect the text column in {str_cols}; set IV_TEXT_COL in config")
    category_col = next((c for c in ["task_name", "scenario", "task", "audio_type", "domain"] if c in str_cols), None)

    print("\n================ IndicVoices: what is in the data ================")
    print("all columns     :", schema.names)
    print("audio column    :", audio_col)
    print("text column     :", text_col, "  <- set IV_TEXT_COL in config(_local).py if another column is better")
    print("category column :", category_col)

    peek = pd.concat([b.to_pandas() for _, b in zip(range(4), pf.iter_batches(500, columns=str_cols))])
    print("\n--- 3 example rows (all text columns) ---")
    for _, r in peek.head(3).iterrows():
        for c in str_cols:
            print(f"{c:>22}: {str(r[c])[:150]}")
        print()
    print("--- short-list columns (candidates for IV_KEEP) ---")
    for c in str_cols:
        vc = peek[c].value_counts()
        if 1 < len(vc) <= 30:
            print(f"\n[{c}]\n{vc.to_string()}")

    rows = next(pf.iter_batches(300, columns=[audio_col, text_col])).to_pylist()
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
        sf.write(ex / f"indicvoices_{i}_{sr}Hz.wav", y, K.SR)
        sf.write(ex / f"indicvoices_{i}_phone_sim.wav", K.phone_sim(y, rng), K.SR)
        (ex / f"indicvoices_{i}.txt").write_text(str(r[text_col]), encoding="utf-8")
    print(f"\nexample audio written to {ex} (original + phone-simulated): listen to them!")
    print("==================================================================\n")
    return audio_col, text_col, category_col


def gen_rows(paths, audio_col, text_col, source, max_hours, keep=None, category_col=None, augment=1, cfg=""):
    tokenizer = K.load_tokenizer()
    total = 0.0
    for y, sr, text, cat in K.iter_clips(paths, audio_col, text_col, keep, category_col):
        t, ids = K.check_clip(tokenizer, text, len(y) / K.SR)
        if t is None:
            DROPS[f"{source}: {ids}"] += 1
            continue
        yield {"audio": K.to_int16(y), "text": t, "labels": ids, "length": len(y), "native_sr": sr,
               "source": source, "category": cat, "augment": augment}
        total += len(y) / K.SR
        if total >= max_hours * 3600:
            return


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
    rows = []
    for y, sr, text, _ in K.iter_clips(paths, audio_col, text_col):
        t, _ = K.check_clip(tokenizer, text, len(y) / K.SR)
        if t is not None:
            rows.append((y, t, sr))
    return rows


def sample_rows(rows, n, seed, phone, exclude=()):
    """n random clips; phone=True -> phone-simulate with a fixed seed per clip (identical every run)."""
    rng, exclude = np.random.default_rng(seed), set(exclude)
    idx = [int(i) for i in rng.permutation(len(rows)) if int(i) not in exclude][:n]
    out = []
    for i in idx:
        y, t, sr = rows[i]
        if phone and sr > 8000:
            y = K.phone_sim(y, np.random.default_rng(seed * 100003 + i))
        out.append((y, t, sr))
    return out, idx


def save_rows(tokenizer, rows, path, source):
    from datasets import Dataset
    Dataset.from_dict({
        "audio": [K.to_int16(y) for y, _, _ in rows], "text": [t for _, t, _ in rows],
        "labels": [tokenizer(t, add_special_tokens=False)["input_ids"] + [K.EOS_ID] for _, t, _ in rows],
        "length": [len(y) for y, _, _ in rows], "native_sr": [sr for _, _, sr in rows],
        "source": [source] * len(rows), "category": [""] * len(rows), "augment": [0] * len(rows),
    }, features=K.features()).save_to_disk(str(path))
    print(f"  {path.name:22s} {len(rows):5d} clips  {sum(len(r[0]) for r in rows) / K.SR / 3600:.2f} h")


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
    real = sorted(C.MY_AUDIO_DIR.glob("*call*")) if C.MY_AUDIO_DIR.is_dir() else []
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

    print("\n[1/5] Download IndicVoices Bengali")
    iv_train_paths = K.download(C.IV_REPO, C.IV_TRAIN_FILES)
    iv_valid_paths = K.download(C.IV_REPO, C.IV_VALID_FILES)
    audio_col, text_col, category_col = inspect_iv(iv_train_paths)
    if args.inspect:
        return

    for d in ("train", "dev", "tests", "_cache"):
        shutil.rmtree(out / d, ignore_errors=True)
    settings_file.unlink(missing_ok=True)
    tokenizer = K.load_tokenizer()
    report = {"iv_columns": {"audio": audio_col, "text": text_col, "category": category_col}}

    print(f"\n[2/5] Build IndicVoices training clips (up to {C.IV_MAX_TRAIN_HOURS} h)")
    DROPS.clear()
    t0 = time.time()
    iv_train = build("iv_train", paths=iv_train_paths, audio_col=audio_col, text_col=text_col, source="indicvoices",
                     max_hours=C.IV_MAX_TRAIN_HOURS, keep=C.IV_KEEP, category_col=category_col)
    print(f"  built in {(time.time() - t0) / 60:.1f} min")
    print(summary(iv_train, "source"))
    if category_col:
        print(summary(iv_train, "category"))
    print(summary(iv_train, "native_sr"))
    print("  dropped:", dict(DROPS))
    report["dropped"] = dict(DROPS)
    parts = [iv_train]

    print("\n[3/5] Clean mix: SUBAK.KO")
    if C.CLEAN_MIX_RATIO > 0:
        target = hours(iv_train) * C.CLEAN_MIX_RATIO / (1 - C.CLEAN_MIX_RATIO)
        print(f"  target {target:.1f} h ({C.CLEAN_MIX_RATIO:.0%} of the training set)")
        sk_paths = K.download(C.SUBAKKO_REPO, C.SUBAKKO_FILES)
        import pyarrow.parquet as pq
        DROPS.clear()
        clean = build("subakko", paths=sk_paths, audio_col=K.find_audio_col(pq.ParquetFile(sk_paths[0]).schema_arrow),
                      text_col="transcription", source="subakko_clean", max_hours=target, augment=2)
        print(summary(clean, "source"))
        print("  dropped:", dict(DROPS))
        report["dropped"].update(DROPS)
        if hours(clean) < 0.9 * target:
            print("  NOTE: fewer clean hours than the target; add more SUBAKKO_FILES for an exact ratio")
        parts.append(clean)
    train = concatenate_datasets(parts).shuffle(seed=C.SEED)
    train.save_to_disk(str(out / "train"))
    report["train"] = summary(train, "source").reset_index().to_dict("records")
    report["train_by_native_sr"] = summary(train, "native_sr").reset_index().to_dict("records")
    report["train_hours"] = round(hours(train), 2)
    report["train_clips"] = len(train)
    print(f"  TRAIN: {len(train)} clips, {hours(train):.1f} h -> {out / 'train'}")
    del train, parts, iv_train
    shutil.rmtree(out / "_cache", ignore_errors=True)

    print("\n[4/5] Dev + test sets")
    iv_valid = load_rows(tokenizer, iv_valid_paths, audio_col, text_col)
    print(f"  IndicVoices valid: {len(iv_valid)} usable clips")
    dev_rows, dev_idx = sample_rows(iv_valid, C.DEV_CLIPS, seed=1, phone=True)
    ivt_rows, _ = sample_rows(iv_valid, C.TEST_CLIPS_PER_SPLIT, seed=1, phone=True, exclude=dev_idx)
    del iv_valid
    save_rows(tokenizer, dev_rows, out / "dev", "dev_iv_phone")

    bs_files = {f.split("/")[-1].split("-")[0]: f for f in list_repo_files(C.BANSPEECH_REPO, repo_type="dataset")
                if f.startswith("data/") and f.endswith(".parquet")}
    cache = {}

    def banspeech(splits, phone, seed=7):
        rows = []
        for k, s in enumerate(splits):
            if s not in cache:
                cache[s] = load_rows(tokenizer, K.download(C.BANSPEECH_REPO, [bs_files[s]]), "audio", "transcription")
            rows += sample_rows(cache[s], C.TEST_CLIPS_PER_SPLIT, seed=seed + k, phone=phone)[0]
        return rows

    tests = {
        "test_iv_phone": ivt_rows,
        "test_bs_spont_phone": banspeech(C.BANSPEECH_SPONT, phone=True),
        "test_bs_spont_clean": banspeech(C.BANSPEECH_SPONT, phone=False),
        "test_bs_clean": banspeech(C.BANSPEECH_CLEAN, phone=False),
    }
    for name, rows in tests.items():
        save_rows(tokenizer, rows, out / "tests" / name, name)
    report["tests"] = {k: len(v) for k, v in tests.items()}

    print("\n[5/5] Phone simulation example")
    phone_sim_picture(tests["test_bs_spont_clean"], out / "examples")

    (out / "data_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    settings_file.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")   # marks "done"
    print(f"\nData ready in {out} ({(time.time() - t_start) / 60:.1f} min). Report: {out / 'data_report.json'}")


def parser():
    p = K.base_parser("Step 1: download + prepare data")
    p.add_argument("--inspect", action="store_true", help="only download + show what is in IndicVoices, then stop")
    p.add_argument("--force", action="store_true", help="rebuild even if prepared data exists")
    return p


if __name__ == "__main__":
    a = parser().parse_args()
    K.setup(a, "prepare_data")
    main(a)
