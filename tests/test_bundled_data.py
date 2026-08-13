"""Guards a real failure mode found by hand: the bundled LoRA fine-tuning
sample was written in ALL CAPS (73.5% uppercase letters) against a
pretraining sample that's almost entirely lowercase (2.2%). LoRA here only
adapts attention Q/V projections, the token embeddings and output head stay
frozen, both shaped almost entirely by the pretraining text's character
distribution. A fine-tune corpus the frozen embeddings/head have barely seen
is a corpus LoRA can't shift *into* no matter how well it trains, so the
"compare" step's output was closer to noise than to a visible style shift.
Confirmed by hand: lowercasing the same fine-tune text, no other change,
dropped validation loss from 3.35 to 2.24 under identical settings.
"""

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def _letter_case_profile(text: str) -> tuple[int, int]:
    letters = [c for c in text if c.isalpha()]
    upper = sum(1 for c in letters if c.isupper())
    return upper, len(letters)


def test_bundled_samples_have_similar_case_distribution():
    base_text = (DATA_DIR / "sample_corpus.txt").read_text()
    ft_text = (DATA_DIR / "finetune_sample.txt").read_text()

    base_upper, base_total = _letter_case_profile(base_text)
    ft_upper, ft_total = _letter_case_profile(ft_text)
    base_pct = base_upper / base_total
    ft_pct = ft_upper / ft_total

    # A generous band, not a tight match: the point is catching another
    # ALL-CAPS-vs-prose mismatch, not enforcing a specific style.
    assert abs(base_pct - ft_pct) < 0.25, (
        f"pretrain sample is {base_pct:.1%} uppercase, fine-tune sample is {ft_pct:.1%} uppercase — "
        "too different for LoRA (which never touches the token embeddings or output head) to bridge"
    )
