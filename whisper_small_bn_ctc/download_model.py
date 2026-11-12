"""
Download Noobbbbb/whisper-small-bn-ctc (Whisper-small encoder + CTC head) into ./model

Usage:
    python download_model.py            # download (skips files already present)
    python download_model.py --force    # re-download
    python download_model.py --token hf_xxx   # only needed for gated repos (or set HF_TOKEN)
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


def load_dotenv(path: Path) -> None:
    """Read KEY=VALUE lines from the repo's .env into os.environ (existing vars win).
    Tiny parser so no extra dependency is needed; values are never printed."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            os.environ.setdefault(key, value)


load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ----------------------------- model settings ------------------------------ #
REPO_ID = "Noobbbbb/whisper-small-bn-ctc"
# ctc_inference.py is NOT downloaded: we use our own reviewed copy (whisper_ctc.py)
ALLOW = ["*.json", "*.safetensors", "README.md"]
REQUIRED = ["ctc_config.json", "encoder_config.json", "encoder.safetensors",
            "head.safetensors", "vocab.json", "preprocessor_config.json"]
GATED = False
# --------------------------------------------------------------------------- #

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
    parser.add_argument("--token", default=None, help="Hugging Face token (gated repos only)")
    parser.add_argument("--retries", type=int, default=5, help="retry on network errors (default 5)")
    args = parser.parse_args()

    log = setup_logging()
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    except ImportError:
        log.error("huggingface_hub is not installed. Run: pip install -r ../requirements.txt")
        return 1

    token = args.token or os.environ.get("HF_TOKEN")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Repo      : %s", REPO_ID)
    log.info("Target dir: %s", MODEL_DIR)
    if GATED and not token:
        log.warning("This repo is gated. Open https://huggingface.co/%s, click 'Agree and access "
                    "repository', create a READ token at https://huggingface.co/settings/tokens "
                    "and pass --token hf_xxx (or set HF_TOKEN).", REPO_ID)

    t0 = time.time()
    try:
        for attempt in range(1, args.retries + 1):
            try:
                snapshot_download(repo_id=REPO_ID, local_dir=MODEL_DIR, token=token,
                                  force_download=args.force and attempt == 1, allow_patterns=ALLOW)
                break
            except (GatedRepoError, KeyboardInterrupt):
                raise
            except Exception as e:  # noqa: BLE001 - network drops: wait, then resume
                if attempt == args.retries:
                    raise
                wait = 10 * attempt
                log.warning("Attempt %d/%d failed (%s: %s). Retrying in %ds (download resumes)...",
                            attempt, args.retries, type(e).__name__, e, wait)
                time.sleep(wait)
    except GatedRepoError as e:
        log.error("Access denied (gated repo). Accept the terms on https://huggingface.co/%s "
                  "and pass --token. Details: %s", REPO_ID, e)
        log.debug(traceback.format_exc())
        return 1
    except HfHubHTTPError as e:
        log.error("HTTP error from Hugging Face: %s", e)
        log.debug(traceback.format_exc())
        return 1
    except Exception as e:  # noqa: BLE001
        log.error("Download failed: %s: %s", type(e).__name__, e)
        log.debug(traceback.format_exc())
        return 1

    missing = [f for f in REQUIRED if not (MODEL_DIR / f).exists()]
    size_mb = sum(p.stat().st_size for p in MODEL_DIR.rglob("*")
                  if p.is_file() and ".cache" not in p.parts) / 1e6
    log.info("Model folder: %.0f MB, downloaded in %.0fs", size_mb, time.time() - t0)
    if missing:
        log.error("Missing files after download: %s", missing)
        return 1
    log.info("Done. Next: python test_model.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
