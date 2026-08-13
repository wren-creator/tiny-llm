from model.tokenizer import CharTokenizer


def test_roundtrip():
    tok = CharTokenizer.default()
    text = "Hello, World! 123"
    assert tok.decode(tok.encode(text)) == text


def test_unknown_chars_dropped_silently():
    tok = CharTokenizer(chars=list("abc"))
    assert tok.encode("abcxyz") == [0, 1, 2]


def test_vocab_size_matches_chars():
    tok = CharTokenizer.default()
    assert tok.vocab_size == len(tok.chars)


def test_from_text_builds_vocab_from_seen_chars_only():
    tok = CharTokenizer.from_text("aab")
    assert tok.vocab_size == 2


def test_json_roundtrip_preserves_vocab():
    tok = CharTokenizer.default()
    restored = CharTokenizer.from_json(tok.to_json())
    assert restored.chars == tok.chars
