"""
Runner for Sanjidh090/moonshine-base-bn (Moonshine-Base, 61.5M, BanglaBERT WordPiece vocab).

Follows the model card's inference recipe (beam search + repetition controls), with:
  * local, offline loading (no network)
  * max_new_tokens tied to the audio length, so a decoder that starts repeating
    on noise/silence cannot run away (an autoregressive model risk on call audio)
  * configurable beam size (1 = greedy, fastest)
"""

from pathlib import Path

import numpy as np
import torch

SR = 16000


class MoonshineBN:
    def __init__(self, model_dir, threads: int = 0, beams: int = 5):
        from transformers import AutoModelForSpeechSeq2Seq, AutoTokenizer

        path = Path(model_dir)
        if not (path / "model.safetensors").exists():
            raise FileNotFoundError(f"model.safetensors not found in {path}. Run download_model.py")
        if threads:
            torch.set_num_threads(threads)

        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            str(path), dtype=torch.float32, local_files_only=True).eval()
        self.beams = beams
        # token ids, with the model card's verified defaults as fallback
        self.start_id = self.tokenizer.cls_token_id if self.tokenizer.cls_token_id is not None else 2
        self.eos_id = self.tokenizer.sep_token_id if self.tokenizer.sep_token_id is not None else 3
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

    @torch.inference_mode()
    def transcribe(self, y: np.ndarray) -> str:
        """y: 16 kHz mono float array (the caller chunks long audio)."""
        y = np.asarray(y, dtype=np.float32)
        if len(y) < 1600:  # < 0.1 s
            return ""
        rem = len(y) % 320
        if rem:
            y = np.concatenate([y, np.zeros(320 - rem, dtype=np.float32)])
        dur = len(y) / SR
        # ~1.3 tokens/word and fast Bangla speech is ~3-4 words/s -> 8 tokens/s is generous
        max_tokens = int(dur * 8) + 16
        ids = self.model.generate(
            torch.from_numpy(y).unsqueeze(0),
            max_new_tokens=max_tokens,
            num_beams=self.beams,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            decoder_start_token_id=self.start_id,
            pad_token_id=self.pad_id,
            eos_token_id=self.eos_id,
        )
        return self.tokenizer.decode(ids[0].tolist(), skip_special_tokens=True).strip()
