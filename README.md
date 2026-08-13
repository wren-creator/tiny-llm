# tiny-llm

A from-scratch teaching LLM you can build on a laptop CPU in a few minutes, with a guided web walkthrough that shows every step instead of hiding it behind an API call.

## Why this exists

Most people's whole relationship with AI is a text box that spits out an answer, with everything before that answer treated as magic. That's a bad way to trust, or fear, anything. So instead of writing another explainer about how LLMs work, this is a thing you run: a real transformer, trained from scratch, on your own machine, in a few minutes.

I don't have a computer science degree. What I've got is twenty five years of hands on keyboards, a decade of that on mainframes, and a habit of learning things by taking them apart instead of waiting to be taught the proper way first. This project is that same habit pointed at LLMs: pop the hood, watch it train, watch a LoRA adapter change its behavior with a tiny fraction of the numbers a full retrain would touch, and understand the thing by building it instead of taking someone's word for how it works. That's sovereignty in the small: not depending on a black box to tell you what it's doing.

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

Once the venv is set up, `./start.sh` and `./stop.sh` run the app in the background (PID-tracked, logs to `app.log`, default port 8501, override with `PORT=8888 ./start.sh`), handy for leaving it running without tying up a terminal.

## What the walkthrough covers

1. **Bring your data.** Use the built-in story, or upload your own `.txt` file. A model only knows what's in the text you give it.
2. **Tokenize.** Text becomes numbers via a character-level tokenizer, the simplest kind: one number per character. You see the vocabulary size and a live encode/decode example.
3. **Configure.** A CPU-friendly default (4 layers, 4 heads, 128-wide) with an advanced panel if you want to change the shape of the model.
4. **Pretrain.** Watch the loss curve fall in real time as a randomly-initialized model learns to predict the next character of your text. Try prompting the freshly trained base model when it's done.
5. **Fine-tune with LoRA.** Pick a second, differently-styled text. Instead of retraining the whole model, LoRA freezes it completely and trains a small set of new low-rank adapter matrices next to a few of its layers. The app shows the actual numbers: in the default run, the base model has 834,560 parameters and the LoRA adapter trains about 8,192 of them, under 1%. That comparison is the entire point of the step.
6. **Compare.** Same prompt, two outputs side by side: the untouched base model, and the base model with the adapter layered on top. The style shift is visible even at this scale, since the base model's own weights never moved.
7. **Retrieve, then generate.** A small retrieval-augmented generation (RAG) step: your training text gets split into chunks and indexed by a from-scratch TF-IDF vector database (`engine/retrieval.py`, no external library), a query gets ranked against every chunk by cosine similarity, and the closest chunk gets handed to the model as extra context before it generates. Chunks are sized against the model's context window, and the app shows you exactly which characters the model ends up reading, since anything past `block_size` gets cropped away and pretending otherwise would defeat the point. The retrieval math is real; whether a model this small can actually make use of the extra context is an open, honestly-stated question, see the caveat in the app.

Along the way, step 4 also has a "Look inside: attention" panel: pick a layer and head and see, as a heatmap, exactly how much each token attended to every earlier token, the actual mechanism inside `CausalSelfAttention` made visible instead of implied.

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

# 5. search a text file with the from-scratch TF-IDF vector index
python -m engine.retrieval --input data/raw/sample_corpus.txt --query "lighthouse storm"
```

Scripts are run with `python -m` (e.g. `python -m engine.train`, not `python engine/train.py`) so Python can resolve the `model`/`data`/`engine` packages correctly from the project root.

## Project layout

```
data/           bundled sample texts, prepare.py (text -> token ids)
model/          tokenizer, the GPT architecture, LoRA, config dataclasses
engine/         pretrain loop, LoRA fine-tune loop, sampling, attention extraction, TF-IDF retrieval
configs/        tiny_cpu.yaml (model/training shape), lora_default.yaml (adapter shape)
tests/          pytest suite covering the tokenizer, GPT forward pass, LoRA, and retrieval
app.py          the guided Streamlit walkthrough
```

`model/gpt.py` has no dependency on HuggingFace `transformers`, on purpose. Every line is something you can read and explain, rather than a call into a library that hides the mechanism this project exists to show.

## Pointing it at your own data

Bring your own `.txt` file at either the "bring your data" step (what the base model pretrains on) or the "fine-tune with LoRA" step (what the adapter learns) in the app, or pass `--input` to `data/prepare.py` on the command line. The tokenizer's vocabulary is a fixed set of printable ASCII characters (letters, digits, common punctuation, whitespace), so base text and fine-tune text share one vocabulary even if you swap files between runs. Characters outside that set (accented letters, emoji, non-Latin scripts) are silently dropped, worth knowing before feeding it anything unusual.

## Getting more legible output

Out of the box, sample output looks something like this:

```
The lighthouse keeevepeper d. The sea se da the skea wa s skeper ir dhe away s ghowawas ways.
```

That's not broken, that's the honest ceiling of a toy-sized model on a couple thousand characters of text. The good sign to look for isn't fluency, it's real words and phrases from your training text bleeding through ("lighthouse," "keeper," "the sea"), proof the model is actually learning the corpus, not just emitting noise. If you want to push it toward something more readable without pretending this is more than it is, a few honest knobs:

- **Lower temperature and top-k when sampling.** `--temperature 0.6 --top-k 10` (defaults are 0.8/20) trades variety for grammar. More repetitive, more coherent.
- **More pretraining steps.** Raise the "pretraining steps" slider in the app, or `max_steps` in `configs/tiny_cpu.yaml`, past the 800-step default. The loss curve usually hasn't leveled off yet at that point.
- **More training text.** The bundled sample story is only a couple thousand characters. A model can't generalize past what it's seen, so a longer corpus (a few short stories concatenated, say) beats a short one every time.

None of that changes the ceiling described above, a model this small was never going to write clean prose. It just moves you closer to that ceiling instead of leaving steps on the table.

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

The suite checks the mechanisms this project's claims rest on, not just that the code runs: the tokenizer's encode/decode roundtrip, the GPT forward pass's output shapes and causal masking (a later token can never change an earlier token's output), attention weights summing to 1 and never looking ahead, LoRA's frozen-base/trainable-adapter split and its zero-init-means-no-behavior-change-yet property, batching and the learning-rate schedule, prompt cropping at the context window, and the TF-IDF retrieval index's ranking and edge cases.

## Scaling this up

See `ROADMAP.md`. Short version: same code, bigger numbers, real compute, eventually a real tokenizer and a real base model. Nothing here gets rewritten to scale up, only reconfigured, which was the point of building it this way in the first place.

## License

GPLv3. See `LICENSE`. If you fork this and change it, that version stays open too, same spirit as the rest of the project.
