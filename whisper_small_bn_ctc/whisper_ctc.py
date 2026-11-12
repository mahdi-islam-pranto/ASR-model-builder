"""
Runner for Noobbbbb/whisper-small-bn-ctc: Whisper-small encoder + CTC head (no decoder).

Same logic as the repo's `ctc_inference.py` (Apache-2.0); kept as a local reviewed copy
so no downloaded code is executed. Loads from a local folder only (offline).

Pipeline: wav 16 kHz -> Whisper log-mel (padded to 30 s) -> Whisper encoder (12 layers)
          -> 2-layer transformer CTC head -> greedy CTC (61 symbols: blank, space, 59 Bangla chars)
"""

import json
import math
import re
import unicodedata
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 16000
WINDOW_SEC = 30  # Whisper always works on 30-second windows


def norm_text(t: str) -> str:
    t = unicodedata.normalize("NFC", str(t))
    t = re.sub(r"[^ঀ-৿ ]", " ", t)  # keep Bangla block + spaces only
    return re.sub(r"\s+", " ", t).strip()


class CTCHead(nn.Module):
    def __init__(self, dim, n_vocab, kind="transformer", n_layers=2, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.body = None
        if kind == "transformer":
            layer = nn.TransformerEncoderLayer(dim, 8, dim * 4, dropout, activation="gelu",
                                               batch_first=True, norm_first=True)
            self.body = nn.TransformerEncoder(layer, n_layers, norm=nn.LayerNorm(dim),
                                              enable_nested_tensor=False)
        self.out = nn.Linear(dim, n_vocab)

    def forward(self, x, lengths):
        x = self.drop(self.norm(x))
        if self.body is not None:
            pad = torch.arange(x.shape[1], device=x.device)[None, :] >= lengths[:, None]
            x = self.body(x, src_key_padding_mask=pad)
        return F.log_softmax(self.out(x), dim=-1)


class WhisperCTC:
    def __init__(self, model_dir, threads: int = 0):
        from safetensors.torch import load_file
        from transformers import WhisperConfig, WhisperFeatureExtractor
        from transformers.models.whisper.modeling_whisper import WhisperEncoder

        path = Path(model_dir)
        if not (path / "encoder.safetensors").exists():
            raise FileNotFoundError(f"encoder.safetensors not found in {path}. Run download_model.py")
        if threads:
            torch.set_num_threads(threads)

        self.cfg = json.loads((path / "ctc_config.json").read_text(encoding="utf-8"))
        self.vocab = json.loads((path / "vocab.json").read_text(encoding="utf-8"))

        enc = WhisperEncoder(WhisperConfig.from_json_file(str(path / "encoder_config.json")))
        if self.cfg.get("encoder_was_cut"):  # upper layers removed, and with them the final norm
            enc.layer_norm = nn.Identity()
        enc.load_state_dict(load_file(str(path / "encoder.safetensors")))
        self.encoder = enc.float().eval()

        self.head = CTCHead(self.cfg["dim"], len(self.vocab), self.cfg["head"], self.cfg["head_layers"])
        self.head.load_state_dict(load_file(str(path / "head.safetensors")))
        self.head = self.head.float().eval()
        self.fe = WhisperFeatureExtractor.from_pretrained(str(path), local_files_only=True)

    @torch.inference_mode()
    def transcribe(self, y: np.ndarray) -> str:
        """y: 16 kHz mono float array, at most 30 s (the caller chunks longer audio)."""
        y = np.asarray(y, dtype=np.float32)
        if len(y) < 400:
            return ""
        feats = self.fe([y], sampling_rate=SR, return_tensors="pt").input_features
        n = min(1500, math.ceil(len(y) / SR * 50))  # encoder frames that hold real audio
        h = self.encoder(feats).last_hidden_state[:, :n].float()
        ids = self.head(h, torch.tensor([n])).argmax(-1)[0].numpy()
        keep = ids[np.insert(ids[1:] != ids[:-1], 0, True)]  # merge repeats, then drop blanks
        return norm_text("".join(self.vocab[i] for i in keep if i != 0))
