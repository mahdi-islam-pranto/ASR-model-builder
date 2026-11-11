"""
Download hishab/titu_stt_bn_fastconformer from Hugging Face into ./model

Usage:
    python download_model.py            # download (skips if already present)
    python download_model.py --force    # re-download
"""

import argparse
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Xet (HF's newer transfer backend) can hang at 0 bytes; plain HTTP is reliable
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

REPO_ID = "hishab/titu_stt_bn_fastconformer"
FILES = [
    "titu_stt_bn_fastconformer.nemo",  # model weights + config + tokenizer
    "test_bn_fastconformer.wav",             # sample clip shipped with the model
]

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model"
LOG_DIR = HERE / "outputs" / "logs"


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"download_{datetime.now():%Y%m%d_%H%M%S}.log"
    logger = logging.getLogger("download")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding="utf-8")):
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.info("Log file: %s", log_file)
    return logger


def main() -> int:
    parser = argparse.ArgumentParser(description=f"Download {REPO_ID}")
    parser.add_argument("--force", action="store_true", help="re-download even if files exist")
    args = parser.parse_args()

    log = setup_logging()
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        log.error("huggingface_hub is not installed. Run: pip install -r ../requirements.txt")
        return 1

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Repo      : %s", REPO_ID)
    log.info("Target dir: %s", MODEL_DIR)

    failed = 0
    for filename in FILES:
        dest = MODEL_DIR / filename
        if dest.exists() and not args.force:
            log.info("[skip] %s already exists (%.1f MB)", filename, dest.stat().st_size / 1e6)
            continue
        log.info("[download] %s ...", filename)
        t0 = time.time()
        try:
            path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                local_dir=MODEL_DIR,
                force_download=args.force,
            )
            size_mb = Path(path).stat().st_size / 1e6
            log.info("[ok] %s  %.1f MB in %.1fs", filename, size_mb, time.time() - t0)
        except Exception as e:  # noqa: BLE001 - we want to log everything
            failed += 1
            log.error("[failed] %s: %s", filename, e)
            log.debug(traceback.format_exc())

    if failed:
        log.error("%d file(s) failed to download. See the log above.", failed)
        return 1
    log.info("Done. Next: python test_model.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
