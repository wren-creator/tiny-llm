from data.prepare import prepare_data
from model.tokenizer import CharTokenizer


def test_split_ratio():
    text = "x" * 100
    train_ids, val_ids, _ = prepare_data(text, val_fraction=0.2)
    assert len(train_ids) == 80
    assert len(val_ids) == 20


def test_default_tokenizer_used_when_none_given():
    _, _, tok = prepare_data("hello")
    assert tok.vocab_size == len(tok.chars) > 0


def test_reuses_given_tokenizer_instead_of_building_a_new_one():
    tok = CharTokenizer.default()
    _, _, returned = prepare_data("hello", tokenizer=tok)
    assert returned is tok
