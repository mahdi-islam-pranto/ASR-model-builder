"""
Download AI4Bharat IndicConformer-600M (multilingual) for Bengali into ./model/<variant>

Two variants of the SAME model (MIT license):
  int8  (default)  yashwork-byte/indic-conformer-600m-int8-onnx
                   community int8-quantized encoder, ~0.8 GB, NOT gated, faster on CPU
  fp32             ai4bharat/indic-conformer-600m-multilingual
                   official files, ~2.6 GB, GATED: needs a Hugging Face token
                   1) open the model page and click "Agree and access repository"
                   2) create a READ token at https://huggingface.co/settings/tokens
                   3) pass --token hf_xxx  (or set the HF_TOKEN env var)

Only the Bengali language head is downloaded (the other 21 languages are skipped).

Usage:
    python download_model.py                       # int8
    python download_model.py --variant fp32 --token hf_xxx
    python download_model.py --force               # re-download
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

VARIANTS = {
    "int8": "yashwork-byte/indic-conformer-600m-int8-onnx",
    "fp32": "ai4bharat/indic-conformer-600m-multilingual",
}
LANG = "bn"
OTHER_LANGS = ["as", "brx", "doi", "gu", "hi", "kn", "kok", "ks", "mai", "ml", "mni",
               "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur"]

HERE = Path(__file__).resolve().parent
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
    parser = argparse.ArgumentParser(description="Download IndicConformer-600M (Bengali)")
    parser.add_argument("--variant", choices=VARIANTS, default="int8")
    parser.add_argument("--token", default=None, help="Hugging Face token (needed for fp32)")
    parser.add_argument("--force", action="store_true", help="re-download even if files exist")
    args = parser.parse_args()

    log = setup_logging()
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    except ImportError:
        log.error("huggingface_hub is not installed. Run: pip install -r ../requirements.txt")
        return 1

    repo_id = VARIANTS[args.variant]
    target = HERE / "model" / args.variant
    target.mkdir(parents=True, exist_ok=True)
    token = args.token or os.environ.get("HF_TOKEN")

    log.info("Variant   : %s", args.variant)
    log.info("Repo      : %s", repo_id)
    log.info("Target dir: %s", target)
    if args.variant == "fp32" and not token:
        log.warning("fp32 is gated. Without a token the download will fail. See --help.")

    t0 = time.time()
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=target,
            token=token,
            force_download=args.force,
            allow_patterns=["assets/*", "config.json", "README.md"],
            # skip the 21 other language heads, we only need Bengali
            ignore_patterns=[f"assets/joint_post_net_{l}.onnx" for l in OTHER_LANGS],
        )
    except GatedRepoError as e:
        log.error("Access denied (gated model). Accept the terms on "
                  "https://huggingface.co/%s and pass --token. Details: %s", repo_id, e)
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

    assets = target / "assets"
    needed = ["preprocessor.ts", "encoder.onnx", "ctc_decoder.onnx", "vocab.json",
              "language_masks.json", "rnnt_decoder.onnx", f"joint_post_net_{LANG}.onnx"]
    missing = [n for n in needed if not (assets / n).exists()]
    size_mb = sum(p.stat().st_size for p in target.rglob("*") if p.is_file()) / 1e6
    log.info("Downloaded %.0f MB in %.0fs", size_mb, time.time() - t0)
    if missing:
        log.error("Missing files after download: %s", missing)
        return 1
    log.info("Done. Next: python test_model.py --variant %s", args.variant)
    return 0


if __name__ == "__main__":
    sys.exit(main())
