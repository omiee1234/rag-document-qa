from ragqa.present import confidence_level, highlight, key_sentences, split_sentences, to_markdown


class _Named:
    def __init__(self, name):
        self.name = name


def test_confidence_levels_use_each_retrievers_own_scale():
    hybrid, tfidf = _Named("hybrid"), _Named("tfidf")
    assert confidence_level(hybrid, 0.9) == "high"
    assert confidence_level(hybrid, 0.05) == "medium"
    assert confidence_level(hybrid, 0.001) == "low"
    assert confidence_level(tfidf, 0.45) == "high"
    assert confidence_level(tfidf, 0.25) == "medium"
    assert confidence_level(tfidf, 0.1) == "low"


def test_key_sentences_are_verbatim_and_in_original_order():
    text = (
        "The company was founded in 2010 by two engineers. "
        "Employees get twenty days of annual leave every year. "
        "The head office moved to Guwahati in 2016. "
        "Unused annual leave can be carried over for one year."
    )
    key = key_sentences("How much annual leave do employees get?", text)
    # the two leave sentences, verbatim, in document order
    assert key == (
        "Employees get twenty days of annual leave every year. "
        "Unused annual leave can be carried over for one year."
    )


def test_key_sentences_prefers_the_retrievers_own_scorer():
    class Scorer:
        def score_passages(self, query, passages):
            return [1.0 if "Guwahati" in p else 0.0 for p in passages]

    text = "Sentence number one is here. The head office is in Guwahati now. A third sentence goes here."
    assert key_sentences("where", text, Scorer(), max_sentences=1) == "The head office is in Guwahati now."


def test_short_chunk_is_returned_whole():
    assert key_sentences("leave", "Employees get twenty days of leave.") == "Employees get twenty days of leave."


def test_highlight_bolds_question_words_including_plural_forms():
    md = highlight("Insured Person's Name: A. Sample", "who are the insured persons")
    assert "**Insured**" in md and "**Person**" in md
    assert "**Name**" not in md


def test_highlight_escapes_markdown_in_source_text():
    md = highlight("*2800000058* # not a heading", "policy")
    assert md.startswith("\\*2800000058\\*") and "\\#" in md


def test_short_lines_render_as_a_list_long_text_as_paragraphs():
    rows = "Gender: Male\nAge: 45\nRelationship: Self"
    assert to_markdown(rows).splitlines() == ["- Gender: Male", "- Age: 45", "- Relationship: Self"]
    para = "A long paragraph. " * 20
    assert not to_markdown(para).startswith("- ")


def test_list_marker_stripping_keeps_words_starting_with_o():
    # a bullet-stripping bug would turn "of the policy" into "f the policy"
    assert "- of the policy" in to_markdown("of the policy\none more line\nother line")


def test_sentence_split_handles_newlines_and_abbreviation_like_text():
    assert split_sentences("First line\nSecond line. Third one!") == ["First line", "Second line.", "Third one!"]
