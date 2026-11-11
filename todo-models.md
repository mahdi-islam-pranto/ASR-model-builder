I've put together a shortlist below. One caveat first: every model card reports WER on different test sets, so the numbers are a rough guide only. The only reliable comparison is to run the shortlisted models on your own Bangla audio, on your CPU.

## What makes a model fast on CPU

- **CTC / Conformer models** (NeMo, wav2vec2, Zipformer) process the whole clip in a single pass. They are usually 3–10× faster on CPU than Whisper at similar accuracy, and they export cleanly to ONNX with int8 quantization.
- **Whisper models** generate text one token at a time, so they are slower on CPU. They can also hallucinate on silence or noise. faster-whisper with int8 makes them usable, but they will still be the slowest option here.

## Candidate models

| # | Model | Type / size | Reported Bangla accuracy | CPU speed | License | Fine-tuning |
|---|---|---|---|---|---|---|
| 1 | **[hishab/titu_stt_bn_conformer_large](https://huggingface.co/hishab/titu_stt_bn_conformer_large)** | NeMo Conformer-CTC Large (~120M) | **Average 7.9% WER / 1.75% CER** over 55h of test data. Fleurs 10.5%, Common Voice 6.5%, Kathbath 9.8% | ⚡ Fast (CTC, ONNX-exportable) | ⚠️ **CC-BY-NC-4.0** (non-commercial) | Easy (NeMo) |
| 2 | **hishab/titu_stt_bn_fastconformer_large_od** | NeMo FastConformer Large | **Average 7.8% WER / 1.68% CER**. Common Voice 6.2%, Fleurs 11.0% | ⚡⚡ Faster than #1 | ⚠️ CC-BY-NC-4.0 | Easy (NeMo) |
| 3 | **[AI4Bharat IndicConformer, Bengali model](https://github.com/AI4Bharat/IndicConformerASR)** (`indicconformer_stt_bn_hybrid_rnnt_large`) | NeMo hybrid RNNT/CTC Large (~120M) | No official Bengali WER published. The related IndicWhisper scored about 11% on Fleurs, and a community ONNX build claims about 8–12% on clean speech | ⚡ Fast (CTC head is ONNX-exportable) | ✅ **MIT** | Easy, but needs AI4Bharat's NeMo fork |
| 4 | [ai4bharat/indic-conformer-600m-multilingual](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual) | 600M, 22 languages | Not published for Bengali | 🐢 Medium (int8 ONNX community builds exist) | MIT (gated download) | Harder (large) |
| 5 | **[bengaliAI/tugstugi_bengaliai-asr_whisper-medium](https://huggingface.co/bengaliAI/tugstugi_bengaliai-asr_whisper-medium)** | Whisper-medium fine-tune (769M) | No WER on the card. Papers report it as the strongest open Whisper for Bangla long-form audio (about 34–38% WER on a hard, noisy Kaggle set) | 🐢 Slow. Needs faster-whisper int8; [a CT2 conversion of a similar model exists](https://huggingface.co/SayedShaun/bengali-whisper-medium-ct2) | ✅ Apache-2.0 | Moderate (HF Transformers) |
| 6 | [bengaliAI regional whisper-medium](https://huggingface.co/bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium) | Whisper-medium | Ranked #1 on the Ben-10 regional-dialect leaderboard | 🐢 Slow | ✅ Apache-2.0 | Moderate |
| 7 | [sherpa-onnx streaming Zipformer bn (vosk, 2026-02)](https://k2-fsa.github.io/sherpa/onnx/pretrained_models) | Small Zipformer transducer | Not published | ⚡⚡⚡ **Fastest**, real-time streaming, runs on any CPU | Probably Apache-2.0 (needs checking) | Possible (icefall/k2), but harder |
| 8 | [arijitx/wav2vec2-xls-r-300m-bengali](https://huggingface.co/arijitx/wav2vec2-xls-r-300m-bengali) | wav2vec2 CTC (300M) | 21.7% WER without a language model, **15.3% WER / 3.4% CER with a 5-gram LM** (OpenSLR-53) | ⚡ OK | Apache-2.0 | Very easy (HF), a good fine-tuning base |

**Not worth testing:**
- Vanilla Whisper large-v3: about 50% WER on Bangla Fleurs, and too heavy for CPU.
- Meta MMS: about 34% WER on Fleurs.
- `bengaliAI/BanglaConformer`: 69% test WER.
- Qwen3-ASR: as far as I know, Bangla isn't in its language list.
- Meta Omnilingual ASR (a 300M CTC int8 version is in sherpa-onnx): it's a broad 1600-language model, so I'd expect it to trail the Bangla-specific models. It's only worth trying as an extra baseline.

## Recommended shortlist

1. **titu_stt_bn_conformer_large / fastconformer_large_od.** These have the best published Bangla numbers, and CTC makes them fast on CPU. The catch is the **non-commercial license**: if your server is for business use, you can test them but not deploy them.
2. **IndicConformer Bengali (MIT).** This is the best commercially safe Conformer option, and it's the base I'd pick for fine-tuning on your own Bangla audio.
3. **bengaliAI whisper-medium via faster-whisper int8.** Use it as an accuracy reference to see whether the slower Whisper is actually more accurate on *your* audio.
4. **sherpa-onnx Bengali Zipformer.** Include it if you need real-time streaming or the lowest CPU use.

All four can be turned into ONNX models served by sherpa-onnx or onnxruntime, which works well for a CPU-only server.

## Things that will change the recommendation

- **Commercial or not?** This decides whether the Hishab models are even an option.
- **Audio type.** Phone or call-center audio (8 kHz) is very different from clean microphone audio. The Hishab Conformer was trained on *non-telephony* data, so phone audio would almost certainly need fine-tuning.
- **Streaming or batch?** Live transcription points to Zipformer or a streaming FastConformer. Transcribing files points to any of the models above.
- **CPU specs** (core count, AVX-512 support), which affect real-time-factor targets.

When you're ready, the next step is a benchmark notebook in [notebooks/](notebooks/). It would run the 3–4 shortlisted models on a Bangla test set (Common Voice / Fleurs bn, plus a sample of your own audio) and report **WER, CER and real-time factor on your CPU**. That will show which model is worth fine-tuning.

Sources:
- [hishab/titu_stt_bn_conformer_large (WER table)](https://huggingface.co/hishab/titu_stt_bn_conformer_large)
- [hishab/titu_stt_bn_fastconformer](https://huggingface.co/hishab/titu_stt_bn_fastconformer)
- [AI4Bharat IndicConformerASR](https://github.com/AI4Bharat/IndicConformerASR)
- [ai4bharat/indic-conformer-600m-multilingual](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual)
- [indic-conformer-600m int8 ONNX](https://huggingface.co/yashwork-byte/indic-conformer-600m-int8-onnx)
- [AI4Bharat Vistaar benchmark](https://github.com/AI4Bharat/vistaar)
- [bengaliAI tugstugi whisper-medium](https://huggingface.co/bengaliAI/tugstugi_bengaliai-asr_whisper-medium)
- [bengaliAI regional whisper-medium](https://huggingface.co/bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium)
- [Ben-10 ASR results](https://huggingface.co/datasets/bengaliAI/ben10-asr-results)
- [SayedShaun/bengali-whisper-medium-ct2](https://huggingface.co/SayedShaun/bengali-whisper-medium-ct2)
- [Bangla long-form ASR paper (arXiv 2603.03158)](https://arxiv.org/html/2603.03158)
- [Bangla long-form ASR paper (arXiv 2602.21183)](https://arxiv.org/html/2602.21183)
- [Wav2Vec2-BERT vs Whisper on Bangla (arXiv 2507.01931)](https://arxiv.org/abs/2507.01931v1)
- [arijitx/wav2vec2-xls-r-300m-bengali](https://huggingface.co/arijitx/wav2vec2-xls-r-300m-bengali)
- [sherpa-onnx pretrained models](https://k2-fsa.github.io/sherpa/onnx/pretrained_models)
- [Meta Omnilingual ASR](https://ai.meta.com/blog/omnilingual-asr-advancing-automatic-speech-recognition/)
- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)
- [Northflank open-source STT 2026](https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2026-benchmarks)