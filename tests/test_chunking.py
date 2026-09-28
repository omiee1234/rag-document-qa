from ragqa.chunking import chunk_text


def test_short_text_produces_one_chunk():
    chunks = chunk_text("doc1", "hello world", chunk_size=800, overlap=150)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "doc1::0"
    assert chunks[0].text == "hello world"


def test_long_text_produces_overlapping_chunks():
    text = "word " * 1000  # 5000 chars
    chunks = chunk_text("doc1", text, chunk_size=800, overlap=150)
    assert len(chunks) > 1
    # ids are sequential and stable
    assert [c.chunk_id for c in chunks] == [f"doc1::{i}" for i in range(len(chunks))]
    # consecutive chunks overlap in character range
    for a, b in zip(chunks, chunks[1:]):
        assert b.start_char < a.end_char


def test_empty_text_produces_no_chunks():
    assert chunk_text("doc1", "   ") == []


def test_chunks_never_start_mid_word():
    # Regression test: found on a real uploaded PDF where a chunk started
    # with "licable" (the tail of "applicable") because the overlap window
    # landed inside a word. end_char is already snapped to whitespace;
    # start_char must be too.
    text = (
        "This clause is applicable on new individual health insurance "
        "policies and not on renewals or at the time of porting or "
        "migrating the policy to another insurer under this agreement. "
    ) * 20
    chunks = chunk_text("doc1", text, chunk_size=200, overlap=50)
    assert len(chunks) > 1

    for chunk in chunks[1:]:  # first chunk always starts at 0, trivially fine
        preceding_char = text[chunk.start_char - 1]
        assert preceding_char.isspace(), (
            f"chunk {chunk.chunk_id!r} starts mid-word: {chunk.text[:30]!r}"
        )
