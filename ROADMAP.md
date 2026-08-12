# Roadmap

The current build is a teaching pipeline: pretrain a tiny GPT from scratch on CPU, then LoRA fine-tune it, both at a scale small enough to watch happen in a browser in a few minutes. Everything below is what changes to move from "understand the mechanism" toward "build something actually useful," without rewriting the mechanism itself.

- [ ] **Real subword tokenizer.** Swap the character-level `CharTokenizer` for a BPE tokenizer (tiktoken or sentencepiece). Same interface (`encode`/`decode`/`vocab_size`), so `model/gpt.py` and the training loops don't change, only `model/tokenizer.py` and the vocab size in configs.
- [ ] **A real domain corpus.** Point the pretraining stage at something bigger than a short story, mainframe and RACF notes, GIBSON teaching material, whatever the actual knowledge base ends up being.
- [ ] **GPU/cloud config.** A larger `GPTConfig` (more layers, wider embeddings, longer context) in a new config file alongside `tiny_cpu.yaml`, sized against whatever hardware gets picked. Use `LLM-calc`'s VRAM/param-count sizing math to pick numbers that actually fit.
- [ ] **LoRA on a real base model.** The realistic route to something domain-useful: point the exact same `model/lora.py` mechanism at a small open-weight base model (a Qwen, Llama, or Mistral variant) instead of a from-scratch pretrain. This is the single biggest capability jump on this list, and it reuses code that already exists rather than requiring new code.
- [ ] **Evaluation.** Wrap the trained/fine-tuned model behind `llm-validator`'s `BaseAdapter` pattern so it can run through that project's accuracy/safety/consistency evaluators instead of eyeballing generated text.
- [ ] **Public deployment.** Once the app is solid, deploy it somewhere public (Streamlit Community Cloud, or self-hosted) so it works as an actual teaching tool for other people, not just a local script.
