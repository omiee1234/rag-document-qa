from ragqa.loaders import _clean_pdf_text


def test_clean_pdf_text_strips_replacement_character():
    dirty = "The insured person shall be allowed �free look period� of 30 days."
    cleaned = _clean_pdf_text(dirty)
    assert "�" not in cleaned
    assert "free look period" in cleaned


def test_clean_pdf_text_strips_private_use_area_bullets():
    # Word-generated PDFs commonly encode Symbol/Wingdings bullets as PUA
    # codepoints (observed in the wild: U+F0B7) rather than U+FFFD -- this
    # is the actual character that showed up in a real uploaded PDF and
    # slipped through the first version of this cleaner, which only
    # stripped U+FFFD.
    dirty = "migrating the policy.\nThe insured person shall be allowed free look"
    cleaned = _clean_pdf_text(dirty)
    assert "" not in cleaned
    assert "The insured person shall be allowed free look" in cleaned


def test_clean_pdf_text_strips_control_characters():
    dirty = "Line one\x0bLine two\x0cLine three"
    cleaned = _clean_pdf_text(dirty)
    assert "\x0b" not in cleaned
    assert "\x0c" not in cleaned


def test_clean_pdf_text_collapses_leftover_extra_spaces():
    dirty = "word1��word2"  # stripping leaves no gap, shouldn't merge words wrongly
    cleaned = _clean_pdf_text(dirty)
    assert cleaned == "word1word2"

    spaced = "word1   word2"
    assert _clean_pdf_text(spaced) == "word1 word2"


def test_clean_pdf_text_preserves_normal_text_and_newlines():
    text = "Normal policy text.\nSecond line with numbers 123 and punctuation, ok."
    assert _clean_pdf_text(text) == text
