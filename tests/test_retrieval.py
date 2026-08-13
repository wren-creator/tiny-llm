from engine.retrieval import TfidfIndex, chunk_text


def test_chunk_text_covers_whole_input():
    text = "a" * 1000
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert chunks[0].startswith("a")
    assert chunks[-1].endswith("a")


def test_chunk_text_empty_input():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_search_ranks_relevant_chunk_first():
    chunks = [
        "The lighthouse keeper watched the storm roll in over the sea.",
        "Quarterly revenue grew twelve percent compared to last year.",
        "The keeper trimmed the lamp wick before nightfall.",
    ]
    index = TfidfIndex.build(chunks)
    results = index.search("lighthouse keeper storm", top_k=2)
    assert results
    assert "lighthouse" in results[0][0]


def test_search_scores_are_sorted_descending():
    chunks = ["red fox", "red fox jumps", "blue whale"]
    index = TfidfIndex.build(chunks)
    results = index.search("red fox jumps", top_k=3)
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_search_on_empty_index_returns_empty():
    index = TfidfIndex.build([])
    assert index.search("anything") == []


def test_query_with_no_known_words_returns_nothing():
    index = TfidfIndex.build(["red blue green"])
    assert index.search("xyzzy plugh") == []
