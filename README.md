# tiny-llm

A from-scratch teaching LLM you can build on a laptop CPU in a few minutes, with a guided web walkthrough that shows every step instead of hiding it behind an API call.

## Why this exists

Most people's relationship with AI is a text box that produces an answer, with everything before that answer treated as magic. This project is the opposite bet: if you walk someone through actually building a language model, tokenizing text, watching a loss curve fall, fine-tuning it with LoRA, and seeing exactly how few parameters that fine-tune touches, the magic goes away and an understandable mechanism is left in its place. That's the whole goal. Not a useful model, an understandable one.

Two things are true at once here, and both matter:

1. **This is a real transformer, trained from real gradients, on your real text.** Nothing here is a simulation or a mockup. The GPT in `model/gpt.py` is the same architecture (causal self-attention, feed-forward blocks, LoRA adapters) that powers much larger models, just at a scale small enough to train on a CPU in minutes.
2. **This will not write fluent English.** At this scale, with a few thousand characters of training text, the output is closer to "recognizable structure and style" than "coherent prose." That's expected, and it's explained at every step so nobody comes away thinking a real assistant just got built. Getting something actually useful means bigger data, a real tokenizer, and real compute, all covered in `ROADMAP.md`.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# guided walkthrough (recommended first run)
streamlit run app.py
```

The app works out of the box with a bundled sample story and a bundled sample "shift log" text for the LoRA fine-tuning step, no data of your own required to try it.

## What the walkthrough covers

1. **Bring your data.** Use the built-in story, or upload your own `.txt` file. A model only knows what's in the text you give it.
2. **Tokenize.** Text becomes numbers via a character-level tokenizer, the simplest kind: one number per character. You see the vocabulary size and a live encode/decode example.
3. **Configure.** A CPU-friendly default (4 layers, 4 heads, 128-wide) with an advanced panel if you want to change the shape of the model.
4. **Pretrain.** Watch the loss curve fall in real time as a randomly-initialized model learns to predict the next character of your text. Try prompting the freshly trained base model when it's done.
5. **Fine-tune with LoRA.** Pick a second, differently-styled text. Instead of retraining the whole model, LoRA freezes it completely and trains a small set of new low-rank adapter matrices next to a few of its layers. The app shows the actual numbers: in the default run, the base model has 834,560 parameters and the LoRA adapter trains about 8,192 of them, under 1%. That comparison is the entire point of the step.
6. **Compare.** Same prompt, two outputs side by side: the untouched base model, and the base model with the adapter layered on top. The style shift is visible even at this scale, since the base model's own weights never moved.

## Running it from the command line instead

Every stage the app drives is also a plain script, so you can run the same pipeline without a browser:

```bash
# 1. tokenize a text file into train/val token id arrays
python -m data.prepare --input data/raw/sample_corpus.txt --outdir data/prepared/sample_corpus

# 2. pretrain a base model from scratch
python -m engine.train --config configs/tiny_cpu.yaml --data data/prepared/sample_corpus --out checkpoints/base.pt

# 3. LoRA fine-tune it on a second text
python -m data.prepare --input data/raw/finetune_sample.txt --outdir data/prepared/finetune_sample
python -m engine.finetune_lora --checkpoint checkpoints/base.pt --config configs/lora_default.yaml \
  --data data/prepared/finetune_sample --out checkpoints/adapter.pt

# 4. sample from either
python -m engine.sample --checkpoint checkpoints/base.pt --prompt "The lighthouse"
python -m engine.sample --checkpoint checkpoints/base.pt --adapter checkpoints/adapter.pt --prompt "The lighthouse"
```

Scripts are run with `python -m` (e.g. `python -m engine.train`, not `python engine/train.py`) so Python can resolve the `model`/`data`/`engine` packages correctly from the project root.

## Project layout

```
data/           bundled sample texts, prepare.py (text -> token ids)
model/          tokenizer, the GPT architecture, LoRA, config dataclasses
engine/         pretrain loop, LoRA fine-tune loop, sampling
configs/        tiny_cpu.yaml (model/training shape), lora_default.yaml (adapter shape)
app.py          the guided Streamlit walkthrough
```

`model/gpt.py` has no dependency on HuggingFace `transformers`, on purpose. Every line is something you can read and explain, rather than a call into a library that hides the mechanism this project exists to show.

## Pointing it at your own data

Bring your own `.txt` file at either the "bring your data" step (what the base model pretrains on) or the "fine-tune with LoRA" step (what the adapter learns) in the app, or pass `--input` to `data/prepare.py` on the command line. The tokenizer's vocabulary is a fixed set of printable ASCII characters (letters, digits, common punctuation, whitespace), so base text and fine-tune text share one vocabulary even if you swap files between runs. Characters outside that set (accented letters, emoji, non-Latin scripts) are silently dropped, worth knowing before feeding it anything unusual.

## Scaling this up

See `ROADMAP.md`. Short version: same code, bigger numbers, real compute, eventually a real tokenizer and a real base model. Nothing here gets rewritten to scale up, only reconfigured.
