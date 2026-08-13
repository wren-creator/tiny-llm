"""A guided, click-through walkthrough of building a tiny language model from
scratch, then fine-tuning it with LoRA — the whole point being to show every
step plainly instead of leaving it as a black box.

Run with: streamlit run app.py

Nothing in this file assumes a particular machine or pre-existing data — the
bundled sample texts mean a first-time visitor can go from zero to a trained,
fine-tuned model with no setup beyond installing requirements.txt.
"""

from __future__ import annotations

import copy
from pathlib import Path

import streamlit as st

import matplotlib.pyplot as plt

from data.prepare import prepare_data
from engine.attention import get_attention_maps
from engine.finetune_lora import finetune_lora
from engine.retrieval import TfidfIndex, chunk_text
from engine.sample import generate_text
from engine.train import train as run_pretrain
from model.config import GPTConfig, LoRAConfig, TrainConfig

ROOT = Path(__file__).resolve().parent
SAMPLE_CORPUS = (ROOT / "data" / "raw" / "sample_corpus.txt").read_text()
FINETUNE_SAMPLE = (ROOT / "data" / "raw" / "finetune_sample.txt").read_text()

st.set_page_config(page_title="Build Your Own Tiny LLM", page_icon="🔬", layout="centered")


def init_state() -> None:
    defaults = dict(
        raw_text=None, tokenizer=None, train_ids=None, val_ids=None,
        model=None, model_config=None, base_params=None, base_trained=False,
        adapted_model=None, adapter_params=None, adapter_added=None, adapter_trained=False,
        retrieval_index=None,
    )
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


init_state()

st.title("Build Your Own Tiny LLM")
st.caption(
    "A hands-on walkthrough of how a language model actually gets built: your text in, "
    "a trained model out, no hidden steps. Everything here runs on a laptop CPU in a few "
    "minutes, at a toy scale small enough to watch happen — it will not be a fluent writer, "
    "but every mechanism it uses is the same one real large language models use."
)

# ---------------------------------------------------------------------------
# 1. Bring your data
# ---------------------------------------------------------------------------
st.header("1. Bring your data")
st.write("A language model only knows what's in the text you give it. Start with the built-in story, or bring your own.")

data_choice = st.radio("Training text", ["Use the built-in sample story", "Upload my own .txt file"], key="data_choice")
uploaded_text = None
if data_choice == "Upload my own .txt file":
    uploaded = st.file_uploader("Text file", type=["txt"], key="data_upload")
    if uploaded is not None:
        uploaded_text = uploaded.read().decode("utf-8", errors="ignore")

candidate_text = uploaded_text if data_choice.startswith("Upload") else SAMPLE_CORPUS
if st.button("Use this text", disabled=candidate_text is None):
    st.session_state.raw_text = candidate_text
    st.session_state.tokenizer = None  # force re-tokenize if data changes
    st.session_state.base_trained = False
    st.session_state.adapter_trained = False
    st.session_state.retrieval_index = None  # force re-index if data changes

if st.session_state.raw_text:
    st.success(f"Loaded {len(st.session_state.raw_text):,} characters.")

    # -----------------------------------------------------------------------
    # 2. Tokenize
    # -----------------------------------------------------------------------
    st.header("2. Tokenize")
    st.write(
        "A model does math, not reading — so every character first gets turned into a number. "
        "This is a character-level tokenizer: the simplest kind, one number per letter/digit/punctuation mark."
    )
    if st.session_state.tokenizer is None:
        train_ids, val_ids, tokenizer = prepare_data(st.session_state.raw_text)
        st.session_state.tokenizer = tokenizer
        st.session_state.train_ids = train_ids
        st.session_state.val_ids = val_ids

    tokenizer = st.session_state.tokenizer
    st.write(f"Vocabulary size: **{tokenizer.vocab_size}** unique characters.")
    preview = st.session_state.raw_text[:40].replace("\n", "\\n")
    encoded_preview = tokenizer.encode(st.session_state.raw_text[:40])
    st.code(f'"{preview}"\n\n  -> {encoded_preview}\n\n  -> "{tokenizer.decode(encoded_preview)}"')

    # -----------------------------------------------------------------------
    # 3. Configure
    # -----------------------------------------------------------------------
    st.header("3. Configure the model")
    st.write("Defaults are sized to train on a laptop CPU in a couple of minutes. Feel free to leave them as-is.")
    with st.expander("Advanced settings (optional)"):
        n_layer = st.slider("n_layer — how many transformer blocks stacked", 1, 8, 4)
        n_head = st.slider("n_head — attention heads per block", 1, 8, 4)
        n_embd = st.select_slider("n_embd — width of the model's internal representation", [32, 64, 128, 256], value=128)
        block_size = st.select_slider("block_size — how many characters of context it can see", [32, 64, 128, 256], value=128)
        max_steps = st.slider("pretraining steps", 100, 3000, 800, step=100)

    # -----------------------------------------------------------------------
    # 4. Pretrain
    # -----------------------------------------------------------------------
    st.header("4. Pretrain")
    st.write("This is the model learning from nothing — random weights at step zero, watch the loss fall as it learns to predict the next character.")

    if st.button("Start pretraining", disabled=st.session_state.base_trained):
        model_config = GPTConfig(vocab_size=tokenizer.vocab_size, block_size=block_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd)
        train_config = TrainConfig(max_steps=max_steps)

        progress = st.progress(0.0)
        status = st.empty()
        chart = st.empty()
        history = {"train_loss": [], "val_loss": []}

        def on_step(step: int, train_loss: float, val_loss: float | None) -> None:
            progress.progress(min(1.0, (step + 1) / max_steps))
            if val_loss is not None:
                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                status.text(f"step {step}/{max_steps} — train loss {train_loss:.3f}, val loss {val_loss:.3f}")
                chart.line_chart(history)

        with st.spinner("Training from scratch..."):
            model = run_pretrain(model_config, train_config, st.session_state.train_ids, st.session_state.val_ids, on_step=on_step)

        st.session_state.model = model
        st.session_state.model_config = model_config
        st.session_state.base_params = model.num_params()
        st.session_state.base_trained = True
        st.session_state.adapter_trained = False  # any prior adapter no longer matches
        st.success(f"Done — a {st.session_state.base_params:,}-parameter model trained from scratch.")

    if st.session_state.base_trained:
        st.subheader("Try the base model")
        base_prompt = st.text_input("Prompt", "The lighthouse", key="base_prompt")
        if st.button("Generate", key="gen_base"):
            output = generate_text(st.session_state.model, tokenizer, base_prompt, max_new_tokens=200)
            st.text_area("Output", output, height=140, key="base_output")

        st.subheader("Look inside: attention")
        st.write(
            "This is the mechanism that makes a transformer a transformer: for every token, attention "
            "decides how much to weigh every earlier token when building that token's representation. "
            "Pick a layer and head below and see those weights for yourself, one row per token, showing "
            "what it attended to (it can only look backward and at itself, never ahead, which is why the "
            "chart is always empty above the diagonal)."
        )
        attn_prompt = st.text_input("Prompt to inspect", "The lighthouse keeper", key="attn_prompt")
        n_layer = st.session_state.model_config.n_layer
        n_head = st.session_state.model_config.n_head
        col1, col2 = st.columns(2)
        attn_layer = col1.selectbox("Layer", list(range(n_layer)), key="attn_layer")
        attn_head = col2.selectbox("Head", list(range(n_head)), key="attn_head")
        if st.button("Show attention", key="show_attn"):
            tokens, attn = get_attention_maps(st.session_state.model, tokenizer, attn_prompt)
            weights = attn[attn_layer, attn_head].numpy()

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(weights, cmap="viridis", vmin=0, vmax=1)
            ax.set_xticks(range(len(tokens)), tokens)
            ax.set_yticks(range(len(tokens)), tokens)
            ax.set_xlabel("attending to")
            ax.set_ylabel("token")
            ax.set_title(f"layer {attn_layer}, head {attn_head}")
            st.pyplot(fig)

        # ---------------------------------------------------------------
        # 5. Fine-tune with LoRA
        # ---------------------------------------------------------------
        st.header("5. Fine-tune with LoRA")
        st.write(
            "Full fine-tuning would mean updating every one of this model's weights again. LoRA instead "
            "freezes the base model completely and trains a small set of new, low-rank adapter matrices "
            "next to a few of its layers — far fewer numbers to learn, and the original model never changes."
        )

        ft_choice = st.radio(
            "Fine-tuning text (pick something in a different style than your training text)",
            ["Use the built-in shift-log sample", "Upload my own .txt file"],
            key="ft_choice",
        )
        ft_uploaded_text = None
        if ft_choice == "Upload my own .txt file":
            ft_uploaded = st.file_uploader("Fine-tuning text file", type=["txt"], key="ft_upload")
            if ft_uploaded is not None:
                ft_uploaded_text = ft_uploaded.read().decode("utf-8", errors="ignore")
        ft_text = ft_uploaded_text if ft_choice.startswith("Upload") else FINETUNE_SAMPLE

        with st.expander("Advanced LoRA settings (optional)"):
            lora_rank = st.slider("rank — width of the adapter", 1, 16, 4)
            lora_steps = st.slider("fine-tuning steps", 50, 1000, 300, step=50)

        if st.button("Start LoRA fine-tuning", disabled=ft_text is None):
            ft_train_ids, ft_val_ids, _ = prepare_data(ft_text, tokenizer=tokenizer)
            lora_config = LoRAConfig(rank=lora_rank, alpha=lora_rank * 2, max_steps=lora_steps)

            progress = st.progress(0.0)
            status = st.empty()
            chart = st.empty()
            history = {"train_loss": [], "val_loss": []}

            def on_step(step: int, train_loss: float, val_loss: float | None) -> None:
                progress.progress(min(1.0, (step + 1) / lora_steps))
                if val_loss is not None:
                    history["train_loss"].append(train_loss)
                    history["val_loss"].append(val_loss)
                    status.text(f"step {step}/{lora_steps} — train loss {train_loss:.3f}, val loss {val_loss:.3f}")
                    chart.line_chart(history)

            # Fine-tune a copy — the base model in session state is left untouched,
            # which is the entire point of LoRA: nothing about it changes.
            base_copy = copy.deepcopy(st.session_state.model)
            with st.spinner("Fine-tuning adapter weights..."):
                adapted_model, added = finetune_lora(base_copy, lora_config, ft_train_ids, ft_val_ids, on_step=on_step)

            st.session_state.adapted_model = adapted_model
            st.session_state.adapter_added = added
            st.session_state.adapter_params = adapted_model.num_trainable_params()
            st.session_state.adapter_trained = True

            pct = 100 * st.session_state.adapter_params / st.session_state.base_params
            st.success(f"Added LoRA adapters to {added} layers.")
            col1, col2 = st.columns(2)
            col1.metric("Base model parameters", f"{st.session_state.base_params:,}")
            col2.metric("Trainable adapter parameters", f"{st.session_state.adapter_params:,}", f"{pct:.2f}% of base")

        # ---------------------------------------------------------------
        # 6. Compare
        # ---------------------------------------------------------------
        if st.session_state.adapter_trained:
            st.header("6. Compare")
            st.write("Same prompt, two models: the untouched base model, and the base model with the LoRA adapter layered on top.")
            compare_prompt = st.text_input("Prompt", "The lighthouse", key="compare_prompt")
            if st.button("Generate both", key="gen_compare"):
                base_out = generate_text(st.session_state.model, tokenizer, compare_prompt, max_new_tokens=180)
                adapted_out = generate_text(st.session_state.adapted_model, tokenizer, compare_prompt, max_new_tokens=180)
                col1, col2 = st.columns(2)
                with col1:
                    st.caption("Base model")
                    st.text_area("Base output", base_out, height=160, key="base_compare_out", label_visibility="collapsed")
                with col2:
                    st.caption("Base + LoRA adapter")
                    st.text_area("Adapted output", adapted_out, height=160, key="adapted_compare_out", label_visibility="collapsed")

        # ---------------------------------------------------------------
        # 7. Retrieve, then generate
        # ---------------------------------------------------------------
        st.header("7. Retrieve, then generate")
        st.write(
            "This is retrieval-augmented generation (RAG): instead of relying only on what the model "
            "learned during training, pull the most relevant snippets of your source text back out at "
            "generation time and hand them to the model as extra context. The retrieval half here is a "
            "small vector database built from scratch (`engine/retrieval.py`), no external library: your "
            "training text is split into overlapping chunks, each chunk becomes a TF-IDF vector (word "
            "counts, weighted down for words that show up everywhere and aren't very informative), and a "
            "query is ranked against every chunk by cosine similarity, the same 'embed things, then find "
            "the nearest ones' idea a production vector database runs, just with an embedding you can "
            "compute by hand instead of a neural one."
        )
        if st.session_state.retrieval_index is None:
            chunks = chunk_text(st.session_state.raw_text)
            st.session_state.retrieval_index = TfidfIndex.build(chunks)
        index = st.session_state.retrieval_index

        retrieval_query = st.text_input("Search your training text for...", "lighthouse", key="retrieval_query")
        if st.button("Search", key="retrieval_search"):
            results = index.search(retrieval_query, top_k=3)
            if not results:
                st.info("No chunk shares any words with that query.")
            else:
                for chunk, score in results:
                    st.markdown(f"**similarity {score:.3f}**")
                    st.text(chunk)

        st.caption(
            "Worth being honest about: a model this small, trained on a couple thousand characters, has "
            "very little capacity to actually *use* extra context the way a real assistant would, it may "
            "just ignore the retrieved text, or fold it in as more raw characters to imitate. The retrieval "
            "mechanism above is real and correct; whether a model is big enough to make good use of what "
            "it retrieves is a separate question, one this scale of model can't really answer."
        )
        rag_prompt = st.text_input("Prompt", "The lighthouse", key="rag_prompt")
        if st.button("Generate with and without retrieved context", key="gen_rag"):
            top_chunks = [chunk for chunk, _ in index.search(rag_prompt, top_k=2)]
            augmented_prompt = "\n\n".join(top_chunks + [rag_prompt]) if top_chunks else rag_prompt

            plain_out = generate_text(st.session_state.model, tokenizer, rag_prompt, max_new_tokens=180)
            augmented_out = generate_text(st.session_state.model, tokenizer, augmented_prompt, max_new_tokens=180)
            col1, col2 = st.columns(2)
            with col1:
                st.caption("Prompt alone")
                st.text_area("Plain output", plain_out, height=160, key="rag_plain_out", label_visibility="collapsed")
            with col2:
                st.caption("Prompt + retrieved context")
                st.text_area("Augmented output", augmented_out, height=160, key="rag_augmented_out", label_visibility="collapsed")

# ---------------------------------------------------------------------------
# What's next
# ---------------------------------------------------------------------------
st.header("What's next?")
st.write(
    "This is a teaching-scale build, not a production one — being honest about that gap is the whole point. "
    "The path from here to something actually useful:"
)
st.markdown(
    """
- Swap the character tokenizer for a real subword (BPE) tokenizer
- Train on a much bigger, domain-specific corpus instead of a short story
- Move `configs/tiny_cpu.yaml` to a larger model and real GPU/cloud compute
- Point the exact same `model/lora.py` mechanism at a real open-weight base model
  (a small Qwen/Llama/Mistral variant) — that's the realistic route to a genuinely useful,
  domain-specific assistant, using the identical fine-tuning code shown above
- See `ROADMAP.md` for the full list
"""
)
