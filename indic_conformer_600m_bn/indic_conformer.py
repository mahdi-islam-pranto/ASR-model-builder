"""
Minimal offline runner for AI4Bharat IndicConformer-600M-multilingual (ONNX export).

Adapted from the model's own `model_onnx.py` (MIT, AI4Bharat) with these changes:
  * loads from a local folder only (no network, no trust_remote_code)
  * loads just the Bengali head instead of all 22 languages (faster start, less RAM)
  * CPU thread count is configurable
  * RNNT parts are only loaded when RNNT decoding is requested

Pipeline:  wav (16 kHz mono) -> preprocessor.ts (log-mel) -> encoder.onnx
           -> CTC:  ctc_decoder.onnx -> greedy collapse
           -> RNNT: rnnt_decoder + joint_* -> greedy transducer search
"""

import json
import warnings
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

BLANK_ID = 256
SOS = 256
RNNT_MAX_SYMBOLS = 10
PRED_RNN_LAYERS = 2
PRED_RNN_HIDDEN_DIM = 640


class IndicConformerONNX:
    def __init__(self, model_dir, lang: str = "bn", threads: int = 0, load_rnnt: bool = True):
        self.assets = Path(model_dir) / "assets"
        if not (self.assets / "encoder.onnx").exists():
            raise FileNotFoundError(f"encoder.onnx not found in {self.assets}. Run download_model.py")
        self.lang = lang

        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._so = so

        with warnings.catch_warnings():  # torch.jit.load deprecation notice is harmless here
            warnings.simplefilter("ignore", FutureWarning)
            self.preprocessor = torch.jit.load(str(self.assets / "preprocessor.ts"), map_location="cpu")
        self.preprocessor.eval()
        self.encoder = self._session("encoder")
        self.ctc_decoder = self._session("ctc_decoder")

        self.vocab = json.loads((self.assets / "vocab.json").read_text(encoding="utf-8"))[lang]
        masks = json.loads((self.assets / "language_masks.json").read_text(encoding="utf-8"))
        self.mask = np.asarray(masks[lang], dtype=bool)

        self.rnnt_loaded = False
        if load_rnnt:
            self.load_rnnt()

    def _session(self, name: str) -> ort.InferenceSession:
        return ort.InferenceSession(str(self.assets / f"{name}.onnx"), sess_options=self._so,
                                    providers=["CPUExecutionProvider"])

    def load_rnnt(self):
        if self.rnnt_loaded:
            return
        self.rnnt_decoder = self._session("rnnt_decoder")
        self.joint_enc = self._session("joint_enc")
        self.joint_pred = self._session("joint_pred")
        self.joint_pre_net = self._session(f"joint_pre_net")
        self.joint_post_net = self._session(f"joint_post_net_{self.lang}")
        self.rnnt_loaded = True

    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def encode(self, wav: np.ndarray):
        """wav: float32 mono 16 kHz numpy array."""
        x = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)).unsqueeze(0)
        feats, length = self.preprocessor(input_signal=x, length=torch.tensor([x.shape[-1]]))
        out, enc_len = self.encoder.run(["outputs", "encoded_lengths"],
                                        {"audio_signal": feats.numpy(), "length": length.numpy()})
        return out, enc_len

    def transcribe(self, wav: np.ndarray, decoding: str = "ctc") -> str:
        enc, enc_len = self.encode(wav)
        if decoding == "ctc":
            return self._ctc(enc)
        if decoding == "rnnt":
            self.load_rnnt()
            return self._rnnt(enc)
        raise ValueError(f"decoding must be 'ctc' or 'rnnt', got {decoding!r}")

    # ------------------------------------------------------------------ #
    def _ctc(self, enc) -> str:
        logits = self.ctc_decoder.run(["logprobs"], {"encoder_output": enc})[0]
        logits = logits[:, :, self.mask]                       # keep Bengali tokens only
        ids = logits[0].argmax(axis=-1)
        # collapse repeats, drop blanks
        keep = np.ones_like(ids, dtype=bool)
        keep[1:] = ids[1:] != ids[:-1]
        ids = ids[keep]
        return "".join(self.vocab[i] for i in ids if i != BLANK_ID).replace("▁", " ").strip()

    def _rnnt(self, enc) -> str:
        f_all = self.joint_enc.run(["output"], {"input": enc.transpose(0, 2, 1)})[0]
        hyp = [SOS]
        state = (np.zeros((PRED_RNN_LAYERS, 1, PRED_RNN_HIDDEN_DIM), dtype=np.float32),
                 np.zeros((PRED_RNN_LAYERS, 1, PRED_RNN_HIDDEN_DIM), dtype=np.float32))
        g_cache = {}

        for t in range(f_all.shape[1]):
            f = f_all[:, t:t + 1, :]
            for _ in range(RNNT_MAX_SYMBOLS):
                key = (hyp[-1], len(hyp))
                if key not in g_cache:  # prediction net only changes when a token is emitted
                    g, _, s0, s1 = self.rnnt_decoder.run(
                        ["outputs", "prednet_lengths", "states", "162"],
                        {"targets": np.array([[hyp[-1]]], dtype=np.int32),
                         "target_length": np.array([1], dtype=np.int32),
                         "states.1": state[0], "onnx::Slice_3": state[1]})
                    g = self.joint_pred.run(["output"], {"input": g.transpose(0, 2, 1)})[0]
                    g_cache = {key: (g, s0, s1)}
                g, s0, s1 = g_cache[key]

                joint = self.joint_pre_net.run(["output"], {"input": f + g})[0]
                logits = self.joint_post_net.run(["output"], {"input": joint})[0]
                tok = int(logits.reshape(-1).argmax())
                if tok == BLANK_ID:
                    break
                hyp.append(tok)
                state = (s0, s1)

        return "".join(self.vocab[i] for i in hyp[1:]).replace("▁", " ").strip()
